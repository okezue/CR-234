use anyhow::{Context, Result};
use clap::Parser;
use csv::WriterBuilder;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parking_lot::Mutex;
use rand::Rng;
use reqwest::header::{HeaderMap, HeaderValue, COOKIE, USER_AGENT};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;
use tokio::time::sleep;

#[derive(Parser)]
#[command(name="cr-scraper",about="Fast Clash Royale 1v1 battle scraper")]
struct Args {
    #[arg(long,default_value="players.csv")]
    players_csv: String,
    #[arg(long,default_value="http://localhost:3000/cf-clearance-scraper")]
    cf_url: String,
    #[arg(long,default_value="")]
    session_cookie: String,
    #[arg(long,default_value_t=32)]
    workers: usize,
    #[arg(long,default_value_t=0)]
    instance_id: usize,
    #[arg(long,default_value_t=1)]
    total_instances: usize,
    #[arg(long,default_value="output")]
    out_dir: String,
    #[arg(long,default_value_t=3)]
    retries: usize,
}

#[derive(Serialize,Deserialize,Clone)]
struct CardPlay {
    battle_id: String,
    x: Option<i32>,
    y: Option<i32>,
    card: String,
    time: i32,
    side: String,
    team: String,
    card_index: Option<i32>,
    level: Option<i32>,
    ability: i32,
    card_type: String,
    player_id: String,
    hero: String,
    evolution: String,
}

#[derive(Serialize,Clone)]
struct BattleMeta {
    replay_tag: String,
    player_id: String,
    timestamp: String,
    team_tags: String,
    opponent_tags: String,
    game_mode: String,
    result: String,
    team_crowns: String,
    opp_crowns: String,
}

#[derive(Deserialize)]
struct CfResponse {
    cookies: Option<Vec<CfCookie>>,
    headers: Option<CfHeaders>,
    source: Option<String>,
}
#[derive(Deserialize)]
struct CfCookie { name: String, value: String, domain: Option<String> }
#[derive(Deserialize)]
struct CfHeaders { #[serde(rename="user-agent")] user_agent: Option<String> }

struct Scraper {
    client: reqwest::Client,
    cf_url: String,
    sem: Arc<Semaphore>,
    scraped: Arc<Mutex<HashSet<String>>>,
    out_dir: PathBuf,
    retries: usize,
    stats: Arc<Mutex<Stats>>,
}

struct Stats { battles: u64, players: u64, plays: u64, skipped_non1v1: u64 }

fn is_1v1(mode: &str) -> bool {
    let m = mode.to_lowercase();
    if m.contains("2v2") || m.contains("triple") || m.contains("mega deck")
        || m.contains("7x") || m.contains("mirror") || m.contains("draft")
        || m.contains("touchdown") || m.contains("boat") || m.contains("war")
        || m.contains("rage") || m.contains("sudden") { return false; }
    if m.is_empty() || m.contains("1v1") || m.contains("ladder")
        || m.contains("path of legends") || m.contains("trophy")
        || m.contains("classic") || m.contains("grand") { return true; }
    true
}

fn detect_card_type(card: &str) -> &'static str {
    let c = card.to_lowercase();
    if c.starts_with("ability") || c.contains("_ability") { return "ability"; }
    if c.ends_with("_ev1") || c.ends_with("_ev2") { return "evolution"; }
    if c.starts_with("hero_") || c.contains("champion") { return "hero"; }
    "normal"
}

fn detect_hero(card: &str) -> String {
    let c = card.to_lowercase();
    if c.starts_with("hero_") || c.contains("champion") { return card.to_string(); }
    String::new()
}

fn detect_evolution(card: &str) -> String {
    let c = card.to_lowercase();
    if c.ends_with("_ev1") || c.ends_with("_ev2") { return card.to_string(); }
    String::new()
}

impl Scraper {
    async fn new(args: &Args) -> Result<Self> {
        let cf_url = args.cf_url.clone();
        let tmp = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;
        let payload = serde_json::json!({"url":"https://royaleapi.com/","mode":"waf-session"});
        let resp: CfResponse = tmp.post(&cf_url)
            .json(&payload).send().await?.json().await?;
        let mut cookie_str = String::new();
        if let Some(cookies) = &resp.cookies {
            for c in cookies {
                if !cookie_str.is_empty() { cookie_str.push_str("; "); }
                cookie_str.push_str(&format!("{}={}", c.name, c.value));
            }
        }
        if !args.session_cookie.is_empty() {
            if !cookie_str.is_empty() { cookie_str.push_str("; "); }
            cookie_str.push_str(&format!("__royaleapi_session_v2={}", args.session_cookie));
        }
        let ua = resp.headers.and_then(|h| h.user_agent)
            .unwrap_or_else(|| "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131".into());
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_str(&ua)?);
        headers.insert(COOKIE, HeaderValue::from_str(&cookie_str)?);
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .gzip(true)
            .build()?;
        let out = PathBuf::from(&args.out_dir);
        fs::create_dir_all(&out)?;
        let mut scraped = HashSet::new();
        let sf = out.join("scraped_players.txt");
        if sf.exists() {
            for line in BufReader::new(fs::File::open(&sf)?).lines().flatten() {
                let p = line.trim().to_string();
                if !p.is_empty() { scraped.insert(p); }
            }
        }
        eprintln!("HTTP client ready. {} previously scraped players.", scraped.len());
        Ok(Self {
            client, cf_url, sem: Arc::new(Semaphore::new(args.workers)),
            scraped: Arc::new(Mutex::new(scraped)), out_dir: out,
            retries: args.retries,
            stats: Arc::new(Mutex::new(Stats { battles: 0, players: 0, plays: 0, skipped_non1v1: 0 })),
        })
    }

    async fn fetch_page(&self, url: &str) -> Result<String> {
        let payload = serde_json::json!({"url": url, "mode": "source"});
        let resp: CfResponse = self.client.post(&self.cf_url)
            .json(&payload).send().await?.json().await?;
        Ok(resp.source.unwrap_or_default())
    }

    async fn fetch_replay(&self, tag: &str) -> Result<String> {
        let url = format!("https://royaleapi.com/data/replay?tag={}", tag);
        let resp = self.client.get(&url).send().await?;
        if !resp.status().is_success() {
            anyhow::bail!("HTTP {}", resp.status());
        }
        let j: serde_json::Value = resp.json().await?;
        if j.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
            Ok(j.get("html").and_then(|v| v.as_str()).unwrap_or("").to_string())
        } else {
            Ok(String::new())
        }
    }

    fn parse_battles_page(&self, html: &str, pid: &str) -> Vec<BattleMeta> {
        let doc = Html::parse_document(html);
        let menu_sel = Selector::parse("div.ui.text.fluid.menu.battle_bottom_menu").unwrap();
        let btn_sel = Selector::parse("button.replay_button").unwrap();
        let ts_sel = Selector::parse("div.item.i18n_duration_short.battle-timestamp-popup").unwrap();
        let gm_sel = Selector::parse(".battle_type, a[href*='/gamemode/']").unwrap();
        let crown_sel = Selector::parse("div.team-segment span.crowns, div.crown-count").unwrap();
        let result_sel = Selector::parse("div.battle_result, span.battle_result").unwrap();
        let mut metas = Vec::new();
        for menu in doc.select(&menu_sel) {
            let btn = match menu.select(&btn_sel).next() { Some(b) => b, None => continue };
            let rid = btn.value().attr("data-replay").unwrap_or("").to_string();
            let team = btn.value().attr("data-team-tags").unwrap_or("").to_string();
            let opp = btn.value().attr("data-opponent-tags").unwrap_or("").to_string();
            let ts = menu.select(&ts_sel).next()
                .and_then(|e| e.value().attr("data-content"))
                .unwrap_or("").to_string();
            let mut gm = String::new();
            let mut el = menu.parent();
            for _ in 0..8 {
                match el {
                    Some(node) => {
                        if let Some(elem) = scraper::ElementRef::wrap(node) {
                            if let Some(g) = elem.select(&gm_sel).next() {
                                gm = g.text().collect::<String>().trim().to_string();
                                if let Some(first) = gm.split('\n').next() { gm = first.trim().to_string(); }
                                break;
                            }
                        }
                        el = node.parent();
                    }
                    None => break,
                }
            }
            let mut result = String::new();
            let mut tc = String::new();
            let mut oc = String::new();
            if let Some(parent) = menu.parent() {
                if let Some(pelem) = scraper::ElementRef::wrap(parent) {
                    if let Some(r) = pelem.select(&result_sel).next() {
                        let txt = r.text().collect::<String>().trim().to_lowercase();
                        if txt.contains("win") || txt.contains("victory") { result = "W".into(); }
                        else if txt.contains("loss") || txt.contains("defeat") { result = "L".into(); }
                        else if txt.contains("draw") { result = "D".into(); }
                    }
                    let crowns: Vec<_> = pelem.select(&crown_sel).collect();
                    if crowns.len() >= 2 {
                        tc = crowns[0].text().collect::<String>().trim().to_string();
                        oc = crowns[1].text().collect::<String>().trim().to_string();
                    }
                }
            }
            metas.push(BattleMeta {
                replay_tag: rid, player_id: pid.to_string(), timestamp: ts,
                team_tags: team, opponent_tags: opp, game_mode: gm,
                result, team_crowns: tc, opp_crowns: oc,
            });
        }
        metas
    }

    fn parse_replay(&self, html: &str, bid: &str, pid: &str) -> Vec<CardPlay> {
        let doc = Html::parse_document(html);
        let marker_sel = Selector::parse("div.markers > div").unwrap();
        let span_sel = Selector::parse("span").unwrap();
        let card_sel = Selector::parse("div.replay_team img.replay_card").unwrap();
        let mut ability_map = std::collections::HashMap::new();
        for img in doc.select(&card_sel) {
            let t = img.value().attr("data-t").unwrap_or("");
            let s = img.value().attr("data-s").unwrap_or("");
            let ab = img.value().attr("data-ability").unwrap_or("");
            if !ab.is_empty() && ab != "None" {
                if let Ok(v) = ab.parse::<i32>() { ability_map.insert((t.to_string(), s.to_string()), v); }
            }
        }
        let mut plays = Vec::new();
        for m in doc.select(&marker_sel) {
            let x = m.value().attr("data-x").and_then(|v| v.parse().ok());
            let y = m.value().attr("data-y").and_then(|v| v.parse().ok());
            let t: i32 = match m.value().attr("data-t").and_then(|v| v.parse().ok()) { Some(v) => v, None => continue };
            let s = m.value().attr("data-s").unwrap_or("").to_string();
            let c = match m.value().attr("data-c") { Some(v) if !v.is_empty() => v.to_string(), _ => continue };
            let idx = m.value().attr("data-i").and_then(|v| v.parse().ok());
            let lvl = m.select(&span_sel).next().map(|sp| sp.text().collect::<String>().trim().to_string())
                .and_then(|v| v.parse().ok());
            let classes = m.value().attr("class").unwrap_or("");
            let team = if classes.contains("red") { "red" } else { "blue" };
            let ab = ability_map.get(&(t.to_string(), s.clone())).copied().unwrap_or(0);
            let ct = detect_card_type(&c);
            let hero = detect_hero(&c);
            let evo = detect_evolution(&c);
            plays.push(CardPlay {
                battle_id: bid.to_string(), x, y, card: c, time: t,
                side: s, team: team.to_string(), card_index: idx,
                level: lvl, ability: ab, card_type: ct.to_string(),
                player_id: pid.to_string(), hero, evolution: evo,
            });
        }
        plays
    }

    async fn scrape_player(self: &Arc<Self>, pid: &str, pb: &ProgressBar) -> Result<(Vec<CardPlay>, Vec<BattleMeta>)> {
        let url = format!("https://royaleapi.com/player/{}/battles/", pid);
        let delay = rand::thread_rng().gen_range(200..800);
        sleep(Duration::from_millis(delay)).await;
        let html = self.fetch_page(&url).await?;
        if html.len() < 1000 { return Ok((vec![], vec![])); }
        let metas = self.parse_battles_page(&html, pid);
        let mut all_plays = Vec::new();
        let mut kept_metas = Vec::new();
        for meta in &metas {
            if !is_1v1(&meta.game_mode) {
                self.stats.lock().skipped_non1v1 += 1;
                continue;
            }
            if meta.replay_tag.is_empty() { continue; }
            let mut rhtml = String::new();
            for attempt in 0..self.retries {
                match self.fetch_replay(&meta.replay_tag).await {
                    Ok(h) if !h.is_empty() => { rhtml = h; break; }
                    Ok(_) => { sleep(Duration::from_millis(500 * (attempt as u64 + 1))).await; }
                    Err(_) => { sleep(Duration::from_millis(1000 * (attempt as u64 + 1))).await; }
                }
            }
            if rhtml.is_empty() { continue; }
            let mut plays = self.parse_replay(&rhtml, &meta.replay_tag, pid);
            let t_unique: HashSet<_> = plays.iter().filter(|p| p.side == "t").map(|p| &p.card).collect();
            let o_unique: HashSet<_> = plays.iter().filter(|p| p.side == "o").map(|p| &p.card).collect();
            if t_unique.len() > 8 || o_unique.len() > 8 { self.stats.lock().skipped_non1v1 += 1; continue; }
            for p in &mut plays {
                if !meta.result.is_empty() {
                    // result already in meta
                }
            }
            kept_metas.push(meta);
            all_plays.extend(plays);
        }
        let s = &self.stats;
        let mut st = s.lock();
        st.battles += kept_metas.len() as u64;
        st.plays += all_plays.len() as u64;
        st.players += 1;
        pb.set_message(format!("{}b {}p {}skip", st.battles, st.players, st.skipped_non1v1));
        Ok((all_plays, kept_metas.into_iter().cloned().collect()))
    }

    async fn run(self: Arc<Self>, player_ids: Vec<String>) -> Result<()> {
        let mp = MultiProgress::new();
        let pb = mp.add(ProgressBar::new(player_ids.len() as u64));
        pb.set_style(ProgressStyle::with_template("{spinner} [{bar:40}] {pos}/{len} {msg}").unwrap());
        let plays_path = self.out_dir.join("card_placements_1v1.csv");
        let meta_path = self.out_dir.join("battle_meta.csv");
        let scraped_path = self.out_dir.join("scraped_players.txt");
        let plays_file = Arc::new(Mutex::new(
            WriterBuilder::new().has_headers(!plays_path.exists())
                .from_writer(OpenOptions::new().create(true).append(true).open(&plays_path)?)
        ));
        let meta_file = Arc::new(Mutex::new(
            WriterBuilder::new().has_headers(!meta_path.exists())
                .from_writer(OpenOptions::new().create(true).append(true).open(&meta_path)?)
        ));
        let scraped_file = Arc::new(Mutex::new(
            OpenOptions::new().create(true).append(true).open(&scraped_path)?
        ));
        let mut handles = Vec::new();
        for pid in player_ids {
            if self.scraped.lock().contains(&pid) { pb.inc(1); continue; }
            let permit = self.sem.clone().acquire_owned().await?;
            let sc = self.clone();
            let pb2 = pb.clone();
            let pf = plays_file.clone();
            let mf = meta_file.clone();
            let sf = scraped_file.clone();
            handles.push(tokio::spawn(async move {
                let _permit = permit;
                match sc.scrape_player(&pid, &pb2).await {
                    Ok((plays, metas)) => {
                        if !plays.is_empty() {
                            let mut w = pf.lock();
                            for p in &plays { let _ = w.serialize(p); }
                            let _ = w.flush();
                        }
                        if !metas.is_empty() {
                            let mut w = mf.lock();
                            for m in &metas { let _ = w.serialize(m); }
                            let _ = w.flush();
                        }
                        {
                            let mut f = sf.lock();
                            let _ = writeln!(f, "{}", pid);
                        }
                    }
                    Err(e) => { eprintln!("Error {}: {}", pid, e); }
                }
                pb2.inc(1);
            }));
        }
        for h in handles { let _ = h.await; }
        pb.finish_with_message("done");
        let st = self.stats.lock();
        eprintln!("Scraped {} battles, {} plays from {} players ({} non-1v1 skipped)",
            st.battles, st.plays, st.players, st.skipped_non1v1);
        Ok(())
    }
}

impl Clone for Scraper {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(), cf_url: self.cf_url.clone(),
            sem: self.sem.clone(), scraped: self.scraped.clone(),
            out_dir: self.out_dir.clone(), retries: self.retries,
            stats: self.stats.clone(),
        }
    }
}

fn load_players(path: &str) -> Result<Vec<String>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut ids = Vec::new();
    for rec in rdr.records() {
        let r = rec?;
        let tag = r.get(0).unwrap_or("").trim().replace('#', "");
        if !tag.is_empty() { ids.push(tag); }
    }
    Ok(ids)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!("Loading players from {}...", args.players_csv);
    let mut ids = load_players(&args.players_csv).context("Failed to load players CSV")?;
    eprintln!("Loaded {} players", ids.len());
    if args.total_instances > 1 {
        let n = ids.len();
        let chunk = n / args.total_instances;
        let start = args.instance_id * chunk;
        let end = if args.instance_id == args.total_instances - 1 { n } else { (args.instance_id + 1) * chunk };
        ids = ids[start..end].to_vec();
        eprintln!("Instance {}/{}: {} players", args.instance_id, args.total_instances, ids.len());
    }
    let scraper = Arc::new(Scraper::new(&args).await?);
    scraper.run(ids).await
}
