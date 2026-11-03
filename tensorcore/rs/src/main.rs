//! tc-hamming — exact pairwise cgMLST Hamming distance matrix on 1-bit tensor
//! cores. Distance = number of loci where both alleles are called and differ
//! (`INF-` stripped; missing char, empty and non-numeric calls = missing).
//!
//! H = C − E, C = P Pᵀ (presence bits), E = O Oᵀ + R: O = one-hot bits of
//! alleles seen ≥ τ times (1-bit AND+popcount GEMMs on the GPU, cuda/tch.cu),
//! R = equal pairs of rarer alleles, enumerated here and subtracted exactly.
//! τ only moves work between the two terms; the result is exact for any τ.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use clap::Parser;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(name = "tc-hamming", version, about = "Exact cgMLST Hamming distance matrix on 1-bit tensor cores")]
struct Args {
    /// Input allele-call TSV (chewBBACA format). Use '-' for stdin
    #[arg(short = 'i', long)]
    input: String,
    /// Output file [default: stdout]
    #[arg(short = 'o', long)]
    output: Option<String>,
    /// Number of CPU threads [default: all cores]
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,
    /// Output matrix layout
    #[arg(short = 'm', long, value_enum, default_value_t = MatrixFormat::Full)]
    matrix_format: MatrixFormat,
    /// Input field separator
    #[arg(short = 'd', long, default_value = "\t")]
    input_sep: String,
    /// Output field separator
    #[arg(short = 'D', long, default_value = "\t")]
    output_sep: String,
    /// Name for the top-left index cell
    #[arg(short = 'x', long, default_value = "tc-hamming")]
    index_name: String,
    /// Character used for missing data
    #[arg(short = 'n', long, default_value = "-")]
    missing_char: String,
    /// Do not strip the chewBBACA `INF-` prefix
    #[arg(short = 'r', long, default_value_t = false)]
    skip_input_replacements: bool,
    /// Keep only loci with at least this % of non-missing calls (0-100)
    #[arg(short = 'L', long)]
    locus_completeness: Option<f64>,
    /// Keep only samples with at least this % of non-missing calls (0-100)
    #[arg(short = 'S', long)]
    sample_completeness: Option<f64>,
    /// gzip-compress the output (also enabled when -o ends with .gz)
    #[arg(short = 'z', long, default_value_t = false)]
    gzip: bool,
    /// Cap distances at this value (0 = no cap)
    #[arg(short = 'X', long, default_value_t = 0)]
    max_dist: i32,
    /// Frequency threshold τ: alleles seen ≥ τ times go to the tensor-core GEMM [0 = auto]
    #[arg(long, default_value_t = 0)]
    tau: usize,
    /// Rows per GPU stripe
    #[arg(long, default_value_t = 512)]
    stripe: usize,
    /// CUDA device
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Suppress progress/log messages on stderr
    #[arg(short = 's', long, default_value_t = false)]
    silent: bool,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq)]
enum MatrixFormat {
    Full,
    #[value(name = "lower-tri")]
    LowerTri,
    #[value(name = "upper-tri")]
    UpperTri,
}

type StripeCb = extern "C" fn(*mut c_void, c_int, c_int, usize, *const u16);
extern "C" {
    fn tch_device_count() -> c_int;
    fn tch_free_vram(device: c_int) -> usize;
    fn tch_run(
        device: c_int, n: c_int, n_pad: usize, o: *const u8, k1: usize, p: *const u8, k2: usize,
        start: *const u32, js: *const u32, n_rare: usize, s: c_int, cb: StripeCb, ctx: *mut c_void,
        gpu_secs: *mut f64, rare_on_gpu: *mut c_int, err: *mut c_char, errlen: c_int,
    ) -> c_int;
}

/// Cell semantics: strip one leading `INF-`, trim,
/// missing char / empty / non-numeric -> 0; integers read as u64.
#[inline]
fn parse_allele(cell: &str, skip_repl: bool, missing: &str) -> Result<u64, ()> {
    let mut s = cell.trim();
    if !skip_repl {
        if let Some(rest) = s.strip_prefix("INF-") {
            s = rest;
        }
    }
    if s == missing || s.is_empty() {
        return Ok(0);
    }
    if s.bytes().all(|b| b.is_ascii_digit()) {
        s.parse::<u64>().map_err(|_| ())
    } else {
        Ok(0)
    }
}

fn read_input(path: &str) -> io::Result<String> {
    let mut buf = String::new();
    if path == "-" {
        io::stdin().read_to_string(&mut buf)?;
    } else {
        File::open(path)?.read_to_string(&mut buf)?;
    }
    Ok(buf)
}

fn up(x: usize, m: usize) -> usize {
    x.div_ceil(m) * m
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

/// Dense per-locus ranks (0 = missing, 1..k) and per-locus allele counts.
fn dense_ranks(calls: &[u64], n: usize, l: usize) -> (Vec<u16>, Vec<Vec<u32>>) {
    let cols: Vec<(Vec<u16>, Vec<u32>)> = (0..l)
        .into_par_iter()
        .map(|k| {
            let col: Vec<u64> = (0..n).map(|i| calls[i * l + k]).collect();
            let mut uniq: Vec<u64> = col.iter().copied().filter(|&v| v != 0).collect();
            uniq.sort_unstable();
            uniq.dedup();
            if uniq.len() > u16::MAX as usize {
                die(format!("locus {} has {} distinct alleles (limit 65535)", k, uniq.len()));
            }
            let mut cnt = vec![0u32; uniq.len() + 1];
            let r: Vec<u16> = col
                .iter()
                .map(|&v| {
                    let x = if v == 0 { 0 } else { uniq.binary_search(&v).unwrap() as u16 + 1 };
                    cnt[x as usize] += 1;
                    x
                })
                .collect();
            (r, cnt)
        })
        .collect();
    let mut x = vec![0u16; n * l];
    x.par_chunks_mut(l.max(1)).enumerate().for_each(|(i, row)| {
        for (k, v) in row.iter_mut().enumerate() {
            *v = cols[k].0[i];
        }
    });
    (x, cols.into_iter().map(|c| c.1).collect())
}

/// τ minimising GEMM time (n²/2 · K1 bits at the measured L4 rate) + rare-pair
/// time, subject to one-hot bits + rare pairs + stripe buffers fitting in VRAM.
fn choose_tau(cnt: &[Vec<u32>], n: usize, stripe: usize, free: usize, silent: bool) -> usize {
    const GEMM_RATE: f64 = 5.9e14; // pair-bits / s
    const RARE_RATE: f64 = 3.5e9; // rare pairs / s
    let c: Vec<u32> = cnt.iter().flat_map(|v| v[1..].iter().copied().filter(|&x| x > 0)).collect();
    let n_pad = up(n, 256);
    let bufs = (stripe * n_pad * 12 + (64 << 20)) as f64;
    let mut best = (0usize, f64::MAX);
    let mut tau = 2;
    while tau <= 4096 {
        let (mut d, mut r) = (0usize, 0f64);
        for &x in &c {
            if x as usize >= tau { d += 1 } else { r += x as f64 * (x as f64 - 1.0) / 2.0 }
        }
        let k1 = up(d.max(1), 1024);
        let mem = (n_pad * k1 / 8) as f64 + 4.0 * (r + n as f64 + 1.0) + bufs;
        let t = (n as f64) * (n as f64) / 2.0 * k1 as f64 / GEMM_RATE + r / RARE_RATE;
        let fits = mem <= free as f64 && r < 4.0e9;
        if !silent {
            eprintln!("  tau={tau:5}  one-hot={d:8}  rare={r:14.0}  est {t:.3}s  vram {:.2} GB{}", mem / 1e9, if fits { "" } else { "  (does not fit)" });
        }
        if fits && t < best.1 {
            best = (tau, t);
        }
        tau *= 2;
    }
    if best.0 == 0 {
        die(format!("no tau fits in {:.2} GB of free VRAM", free as f64 / 1e9));
    }
    best.0
}

struct Sink<'a> {
    upper: &'a mut [Vec<u16>],
}

extern "C" fn on_stripe(ctx: *mut c_void, r0: c_int, rows: c_int, w: usize, stripe: *const u16) {
    let sink = unsafe { &mut *(ctx as *mut Sink) };
    let (r0, rows) = (r0 as usize, rows as usize);
    let stripe = unsafe { std::slice::from_raw_parts(stripe, rows * w) };
    sink.upper[r0..r0 + rows].par_iter_mut().enumerate().for_each(|(r, row)| {
        *row = stripe[r * w + r + 1..(r + 1) * w].to_vec();
    });
}

#[inline]
fn at(upper: &[Vec<u16>], i: usize, j: usize) -> u16 {
    if i == j {
        0
    } else if j > i {
        upper[i][j - i - 1]
    } else {
        upper[j][i - j - 1]
    }
}

/// Header + rows, formatted in parallel chunks and written in order.
fn write_matrix<W: Write>(out: &mut W, ids: &[&str], upper: &[Vec<u16>], osep: &[u8], index_name: &str, mfmt: MatrixFormat, cap: i32) {
    let n = ids.len();
    out.write_all(index_name.as_bytes()).unwrap();
    for id in ids {
        out.write_all(osep).unwrap();
        out.write_all(id.as_bytes()).unwrap();
    }
    out.write_all(b"\n").unwrap();
    const CHUNK: usize = 1024;
    let mut start = 0;
    while start < n {
        let end = (start + CHUNK).min(n);
        let bufs: Vec<Vec<u8>> = (start..end)
            .into_par_iter()
            .map(|i| {
                let mut buf = Vec::with_capacity((n + 1) * 7);
                let mut ib = itoa::Buffer::new();
                buf.extend_from_slice(ids[i].as_bytes());
                for j in 0..n {
                    buf.extend_from_slice(osep);
                    let keep = match mfmt {
                        MatrixFormat::Full => true,
                        MatrixFormat::LowerTri => j <= i,
                        MatrixFormat::UpperTri => j >= i,
                    };
                    let mut v = if keep { at(upper, i, j) as i32 } else { 0 };
                    if cap > 0 && v > cap {
                        v = cap;
                    }
                    buf.extend_from_slice(ib.format(v).as_bytes());
                }
                buf.push(b'\n');
                buf
            })
            .collect();
        for b in &bufs {
            out.write_all(b).unwrap();
        }
        start = end;
    }
}

fn main() {
    let args = Args::parse();
    let t_start = Instant::now();
    let log = |m: String| {
        if !args.silent {
            eprintln!("{m}");
        }
    };
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().ok();
    }
    if args.input_sep.len() != 1 {
        die("input separator must be a single character");
    }
    let sep = args.input_sep.as_bytes()[0] as char;
    if unsafe { tch_device_count() } as usize <= args.device {
        die(format!("CUDA device {} not available", args.device));
    }

    // ---- load ----
    let t = Instant::now();
    let text = read_input(&args.input).unwrap_or_else(|e| die(format!("reading {}: {e}", args.input)));
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_else(|| die("empty input"));
    let mut l = header.split(sep).count() - 1;
    let data: Vec<&str> = lines.filter(|x| !x.is_empty()).collect();
    let mut n = data.len();
    let (skip, missing) = (args.skip_input_replacements, args.missing_char.as_str());
    let parsed: Vec<(&str, Vec<u64>)> = data
        .par_iter()
        .map(|line| {
            let mut it = line.split(sep);
            let id = it.next().unwrap_or("");
            let mut c: Vec<u64> = it
                .map(|cell| parse_allele(cell, skip, missing).unwrap_or_else(|_| die(format!("sample {id}: allele '{}' too large", cell.trim()))))
                .collect();
            c.resize(l, 0);
            (id, c)
        })
        .collect();
    let mut ids: Vec<&str> = parsed.iter().map(|p| p.0).collect();
    let mut calls = vec![0u64; n * l];
    calls.par_chunks_mut(l.max(1)).zip(parsed.par_iter()).for_each(|(dst, p)| dst.copy_from_slice(&p.1[..l]));
    drop(parsed);
    log(format!("Loaded {n} samples x {l} loci in {:.2}s", t.elapsed().as_secs_f64()));

    // ---- completeness filters (present = non-zero) ----
    if let Some(thr) = args.locus_completeness {
        let keep: Vec<usize> = (0..l)
            .into_par_iter()
            .filter(|&k| ((0..n).filter(|&i| calls[i * l + k] != 0).count() as f64 / n as f64) * 100.0 >= thr)
            .collect();
        if keep.len() != l {
            let mut nc = vec![0u64; n * keep.len()];
            nc.par_chunks_mut(keep.len().max(1)).enumerate().for_each(|(i, row)| {
                for (d, &k) in keep.iter().enumerate() {
                    row[d] = calls[i * l + k];
                }
            });
            calls = nc;
            l = keep.len();
        }
        log(format!("Locus completeness >= {thr}%: kept {l} loci"));
    }
    if let Some(thr) = args.sample_completeness {
        let keep: Vec<usize> = (0..n)
            .into_par_iter()
            .filter(|&i| (calls[i * l..(i + 1) * l].iter().filter(|&&v| v != 0).count() as f64 / l as f64) * 100.0 >= thr)
            .collect();
        if keep.len() != n {
            let mut nc = vec![0u64; keep.len() * l];
            let mut nids = Vec::with_capacity(keep.len());
            for (d, &i) in keep.iter().enumerate() {
                nc[d * l..(d + 1) * l].copy_from_slice(&calls[i * l..(i + 1) * l]);
                nids.push(ids[i]);
            }
            calls = nc;
            ids = nids;
            n = keep.len();
        }
        log(format!("Sample completeness >= {thr}%: kept {n} samples"));
    }
    if l > u16::MAX as usize {
        die("more than 65535 loci (u16 distances)");
    }

    // ---- encode ----
    let t = Instant::now();
    let (x, cnt) = dense_ranks(&calls, n, l);
    drop(calls);
    let mut upper: Vec<Vec<u16>> = vec![Vec::new(); n];
    let mut gpu_secs = 0f64;
    if n >= 2 && l > 0 {
        let stripe = args.stripe.max(8) / 8 * 8;
        let tau = if args.tau > 0 {
            args.tau
        } else {
            choose_tau(&cnt, n, stripe, unsafe { tch_free_vram(args.device as c_int) }, args.silent)
        };
        // dense column of (locus, rank) if frequent, else -1
        let mut base = vec![0usize; l + 1];
        let col: Vec<Vec<i32>> = cnt
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let mut v = vec![-1i32; c.len()];
                for a in 1..c.len() {
                    if c[a] as usize >= tau {
                        v[a] = base[k + 1] as i32;
                        base[k + 1] += 1;
                    }
                }
                v
            })
            .collect();
        for k in 0..l {
            base[k + 1] += base[k];
        }
        let d = base[l];
        let n_pad = up(n, 256);
        let (k1, k2) = (up(d.max(1), 1024), up(l, 1024));
        let mut o = vec![0u8; n_pad * k1 / 8];
        let mut p = vec![0u8; n_pad * k2 / 8];
        o.par_chunks_mut(k1 / 8).zip(p.par_chunks_mut(k2 / 8)).take(n).enumerate().for_each(|(i, (orow, prow))| {
            for k in 0..l {
                let a = x[i * l + k] as usize;
                if a == 0 {
                    continue;
                }
                prow[k >> 3] |= 1 << (k & 7);
                let c = col[k][a];
                if c >= 0 {
                    let c = base[k] + c as usize;
                    orow[c >> 3] |= 1 << (c & 7);
                }
            }
        });
        // rare equal pairs (i < j), CSR by i
        let groups: Vec<Vec<Vec<u32>>> = (0..l)
            .into_par_iter()
            .map(|k| {
                let mut g: Vec<Vec<u32>> = vec![Vec::new(); cnt[k].len()];
                for i in 0..n {
                    let a = x[i * l + k] as usize;
                    if a != 0 && col[k][a] < 0 && cnt[k][a] >= 2 {
                        g[a].push(i as u32);
                    }
                }
                g.into_iter().filter(|v| v.len() >= 2).collect()
            })
            .collect();
        let counts: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
        groups.par_iter().flatten().for_each(|v| {
            for (a, &i) in v.iter().enumerate() {
                counts[i as usize].fetch_add((v.len() - 1 - a) as u64, Ordering::Relaxed);
            }
        });
        let mut start = vec![0u32; n + 1];
        let mut n_rare = 0u64;
        for i in 0..n {
            start[i] = n_rare as u32;
            n_rare += counts[i].load(Ordering::Relaxed);
            if n_rare >= 1 << 32 {
                die("too many rare pairs: raise --tau");
            }
        }
        start[n] = n_rare as u32;
        let fill: Vec<AtomicU32> = start[..n].iter().map(|&s| AtomicU32::new(s)).collect();
        let js: Vec<AtomicU32> = (0..n_rare).map(|_| AtomicU32::new(0)).collect();
        groups.par_iter().flatten().for_each(|v| {
            for a in 0..v.len() {
                for &j in &v[a + 1..] {
                    js[fill[v[a] as usize].fetch_add(1, Ordering::Relaxed) as usize].store(j, Ordering::Relaxed);
                }
            }
        });
        drop(groups);
        let js: Vec<u32> = js.into_par_iter().map(|a| a.into_inner()).collect();
        log(format!("Encoded in {:.2}s: tau={tau}, one-hot cols={d}, rare equal pairs={n_rare}", t.elapsed().as_secs_f64()));

        // ---- GPU ----
        let t = Instant::now();
        let mut sink = Sink { upper: &mut upper };
        let mut err = vec![0 as c_char; 512];
        let mut rare_gpu: c_int = 0;
        let rc = unsafe {
            tch_run(
                args.device as c_int, n as c_int, n_pad, o.as_ptr(), k1, p.as_ptr(), k2, start.as_ptr(), js.as_ptr(),
                n_rare as usize, stripe as c_int, on_stripe, &mut sink as *mut Sink as *mut c_void, &mut gpu_secs,
                &mut rare_gpu, err.as_mut_ptr(), err.len() as c_int,
            )
        };
        if rc != 0 {
            die(format!("CUDA: {}", unsafe { std::ffi::CStr::from_ptr(err.as_ptr()) }.to_string_lossy()));
        }
        log(format!(
            "Computed {n} x {n} matrix on tensor cores in {:.2}s (GPU pipeline {:.2}s, rare pairs on {})",
            t.elapsed().as_secs_f64(), gpu_secs, if rare_gpu != 0 { "GPU" } else { "host" }
        ));
    } else {
        for (i, r) in upper.iter_mut().enumerate() {
            *r = vec![0; n - 1 - i];
        }
    }

    // ---- write ----
    let t = Instant::now();
    let osep = args.output_sep.as_bytes().to_vec();
    let gzip = args.gzip || args.output.as_deref().is_some_and(|p| p.ends_with(".gz"));
    let fw: Box<dyn Write + Send> = match &args.output {
        Some(p) => Box::new(File::create(p).unwrap_or_else(|e| die(format!("creating {p}: {e}")))),
        None => Box::new(io::stdout()),
    };
    if gzip {
        use gzp::{deflate::Gzip, par::compress::ParCompressBuilder, ZWriter};
        let mut z = ParCompressBuilder::<Gzip>::new().compression_level(gzp::Compression::new(3)).from_writer(fw);
        write_matrix(&mut z, &ids, &upper, &osep, &args.index_name, args.matrix_format, args.max_dist);
        z.finish().unwrap();
    } else {
        let mut out = BufWriter::with_capacity(16 << 20, fw);
        write_matrix(&mut out, &ids, &upper, &osep, &args.index_name, args.matrix_format, args.max_dist);
        out.flush().unwrap();
    }
    log(format!("Wrote output in {:.2}s. Total {:.2}s", t.elapsed().as_secs_f64(), t_start.elapsed().as_secs_f64()));
}
