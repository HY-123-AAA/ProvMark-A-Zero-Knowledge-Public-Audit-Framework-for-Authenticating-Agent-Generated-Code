// main.rs - ProvMark ZKP circuit (arkworks 0.3 + GM17 + BLS12-381)
// ✅ 修复：解码公式从 r⊕b 改为 g⊕b
// ---------------------------------------------------------------

use anyhow::{bail, Context, Result};
use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::{PrimeField, Zero};
use ark_gm17::{GM17, Proof, ProvingKey, VerifyingKey};
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_r1cs_std::{
    alloc::AllocVar,
    boolean::Boolean,
    fields::fp::FpVar,
    prelude::{EqGadget, FieldVar, ToBitsGadget},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_snark::{CircuitSpecificSetupSNARK, SNARK};
use rand::{rngs::StdRng, SeedableRng};
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

const FIX_BIT_IDX: bool = false;

// =========================
// 工具函数
// =========================
fn enforce_u_n(x: &FpVar<Fr>, n: usize) -> Result<Vec<Boolean<Fr>>, SynthesisError> {
    let bits = x.to_bits_le()?;
    for i in n..bits.len() {
        bits[i].enforce_equal(&Boolean::constant(false))?;
    }
    Ok(bits[..n].to_vec())
}

fn gt_bits_le(a_bits_le: &[Boolean<Fr>], b_bits_le: &[Boolean<Fr>]) -> Result<Boolean<Fr>, SynthesisError> {
    assert_eq!(a_bits_le.len(), b_bits_le.len());
    let n = a_bits_le.len();

    let mut greater = Boolean::constant(false);
    let mut equal = Boolean::constant(true);

    for i in (0..n).rev() {
        let ai = a_bits_le[i].clone();
        let bi = b_bits_le[i].clone();

        let ai_and_not_bi = Boolean::and(&ai, &bi.not())?;
        let g_i = Boolean::and(&equal, &ai_and_not_bi)?;
        greater = Boolean::or(&greater, &g_i)?;

        let neq = Boolean::xor(&ai, &bi)?;
        let eq_i = neq.not();
        equal = Boolean::and(&equal, &eq_i)?;
    }
    Ok(greater)
}

fn or_reduce(bits: &[Boolean<Fr>]) -> Result<Boolean<Fr>, SynthesisError> {
    let mut acc = Boolean::constant(false);
    for b in bits {
        acc = Boolean::or(&acc, b)?;
    }
    Ok(acc)
}

fn parse_bits_string(s: &str) -> Result<Vec<bool>> {
    let mut out = Vec::with_capacity(s.len());
    for (i, ch) in s.chars().enumerate() {
        match ch {
            '0' => out.push(false),
            '1' => out.push(true),
            _ => bail!("invalid bit char at pos {}: {:?}", i, ch),
        }
    }
    Ok(out)
}

// =========================
// JSON 输入结构
// =========================
#[derive(Debug, Deserialize, Clone)]
struct SnarkInput {
    model_id_u64: u64,
    tau_times_1000: u64,
    gamma_times_1000: u64,
    e_max: u64,
    n: usize,
    #[serde(alias = "L")]
    l: usize,
    bits_len: usize,

    m: Vec<u8>,
    h: Vec<u8>,
    g: Vec<Vec<u8>>,
    r: Vec<Vec<u8>>,
    b: Vec<Vec<u8>>,
    bit_idx: Option<Vec<u64>>,
    original_bits: String,

    #[serde(default)]
    decoded_bits: Option<String>,
    #[serde(default)]
    hamming_distance: Option<usize>,
}

impl SnarkInput {
    fn sanity_check(&self) -> Result<()> {
        if self.l == 0 || self.n == 0 || self.bits_len == 0 {
            bail!("l, n, bits_len must be > 0");
        }
        if self.m.len() != self.bits_len {
            bail!("m length must equal bits_len");
        }
        if self.h.len() != self.n {
            bail!("h length must equal n");
        }
        if self.g.len() != self.n || self.r.len() != self.n || self.b.len() != self.n {
            bail!("g/r/b length must equal n");
        }
        for t in 0..self.n {
            if self.g[t].len() != self.l || self.r[t].len() != self.l || self.b[t].len() != self.l {
                bail!("g[{}]/r[{}]/b[{}] length must equal l", t, t, t);
            }
        }
        if self.original_bits.len() != self.bits_len {
            bail!("original_bits length must equal bits_len");
        }

        for t in 0..self.n {
            let ht = self.h[t];
            for i in 0..self.l {
                let sum = self.g[t][i] + self.r[t][i];
                if ht != 0 {
                    if sum != 1 {
                        bail!("h[{}]=1 but g+r != 1 at layer {}", t, i);
                    }
                } else {
                    if sum != 0 {
                        bail!("h[{}]=0 but g+r != 0 at layer {}", t, i);
                    }
                }
            }
        }

        if !FIX_BIT_IDX {
            let bit_idx = self.bit_idx.as_ref()
                .ok_or_else(|| anyhow::anyhow!("bit_idx is required when FIX_BIT_IDX=false"))?;
            if bit_idx.len() != self.n {
                bail!("bit_idx length must equal n");
            }
            for t in 0..self.n {
                if self.h[t] != 0 && bit_idx[t] >= self.bits_len as u64 {
                    bail!("bit_idx[{}]={} out of range while h[t]=1", t, bit_idx[t]);
                }
            }
        }

        if let Some(db) = &self.decoded_bits {
            let db_bits = parse_bits_string(db)?;
            if db_bits.len() != self.bits_len {
                bail!("decoded_bits length must equal bits_len");
            }
            for j in 0..self.bits_len {
                let mj = self.m[j] != 0;
                if mj != db_bits[j] {
                    bail!("m[{}] != decoded_bits[{}]", j, j);
                }
            }
        }

        let orig = parse_bits_string(&self.original_bits)?;
        let mut hd = 0usize;
        for j in 0..self.bits_len {
            let mj = self.m[j] != 0;
            if mj != orig[j] {
                hd += 1;
            }
        }
        if let Some(hd_json) = self.hamming_distance {
            if hd_json != hd {
                bail!("hamming_distance mismatch: json={}, computed={}", hd_json, hd);
            }
        }
        if hd as u64 > self.e_max {
            bail!("HD(m, original_bits)={} > e_max={}", hd, self.e_max);
        }

        Ok(())
    }
}

// =========================
// 电路
// =========================
#[derive(Clone)]
struct ProvMarkCircuit {
    inp: SnarkInput,
}

impl ConstraintSynthesizer<Fr> for ProvMarkCircuit {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let n = self.inp.n;
        let l = self.inp.l;
        let bits_len = self.inp.bits_len;

        // ---- public inputs ----
        let model_id = FpVar::<Fr>::new_input(cs.clone(), || Ok(Fr::from(self.inp.model_id_u64)))?;
        let tau_times_1000 = FpVar::<Fr>::new_input(cs.clone(), || Ok(Fr::from(self.inp.tau_times_1000)))?;
        let gamma_times_1000 = FpVar::<Fr>::new_input(cs.clone(), || Ok(Fr::from(self.inp.gamma_times_1000)))?;
        let e_max = FpVar::<Fr>::new_input(cs.clone(), || Ok(Fr::from(self.inp.e_max)))?;

        let _ = enforce_u_n(&model_id, 64)?;
        let _ = enforce_u_n(&tau_times_1000, 32)?;
        let _ = enforce_u_n(&gamma_times_1000, 32)?;
        let _ = enforce_u_n(&e_max, 32)?;

        // ---- witnesses ----
        let mut m_bits: Vec<Boolean<Fr>> = Vec::with_capacity(bits_len);
        for j in 0..bits_len {
            m_bits.push(Boolean::new_witness(cs.clone(), || Ok(self.inp.m[j] != 0))?);
        }

        let orig_bits_native = parse_bits_string(&self.inp.original_bits)
            .map_err(|_| SynthesisError::AssignmentMissing)?;
        let mut orig_bits: Vec<Boolean<Fr>> = Vec::with_capacity(bits_len);
        for j in 0..bits_len {
            orig_bits.push(Boolean::new_witness(cs.clone(), || Ok(orig_bits_native[j]))?);
        }

        let mut h_bits: Vec<Boolean<Fr>> = Vec::with_capacity(n);
        for t in 0..n {
            h_bits.push(Boolean::new_witness(cs.clone(), || Ok(self.inp.h[t] != 0))?);
        }

        let mut g_matrix: Vec<Vec<Boolean<Fr>>> = Vec::with_capacity(n);
        let mut r_matrix: Vec<Vec<Boolean<Fr>>> = Vec::with_capacity(n);
        let mut b_matrix: Vec<Vec<Boolean<Fr>>> = Vec::with_capacity(n);

        for t in 0..n {
            let mut g_row = Vec::with_capacity(l);
            let mut r_row = Vec::with_capacity(l);
            let mut b_row = Vec::with_capacity(l);
            for i in 0..l {
                g_row.push(Boolean::new_witness(cs.clone(), || Ok(self.inp.g[t][i] != 0))?);
                r_row.push(Boolean::new_witness(cs.clone(), || Ok(self.inp.r[t][i] != 0))?);
                b_row.push(Boolean::new_witness(cs.clone(), || Ok(self.inp.b[t][i] != 0))?);
            }
            g_matrix.push(g_row);
            r_matrix.push(r_row);
            b_matrix.push(b_row);
        }

        let mut bit_idx_vars: Vec<FpVar<Fr>> = Vec::with_capacity(n);
        if !FIX_BIT_IDX {
            let bit_idx = self.inp.bit_idx.clone().unwrap_or_default();
            for t in 0..n {
                let idx = FpVar::<Fr>::new_witness(cs.clone(), || Ok(Fr::from(bit_idx[t])))?;
                let _ = enforce_u_n(&idx, 32)?;
                bit_idx_vars.push(idx);
            }
        }

        // =========================
        // (1) 一致性 + 统计量
        // =========================
        let mut n_valid = FpVar::<Fr>::zero();
        let mut x_green = FpVar::<Fr>::zero();

        for t in 0..n {
            let active_t = h_bits[t].clone();
            n_valid += FpVar::<Fr>::from(active_t.clone());

            for i in 0..l {
                let g_ti = g_matrix[t][i].clone();
                let r_ti = r_matrix[t][i].clone();
                let sum_gr = FpVar::<Fr>::from(g_ti.clone()) + FpVar::<Fr>::from(r_ti.clone());
                sum_gr.enforce_equal(&FpVar::<Fr>::from(active_t.clone()))?;

                x_green += FpVar::<Fr>::from(g_ti);
            }
        }

        let n_valid_bits = enforce_u_n(&n_valid, 32)?;
        let any_valid = or_reduce(&n_valid_bits)?;
        any_valid.enforce_equal(&Boolean::constant(true))?;

        // =========================
        // (2) Z 检验
        // =========================
        let thousand = FpVar::<Fr>::constant(Fr::from(1000u64));
        let million = FpVar::<Fr>::constant(Fr::from(1_000_000u64));
        let l_fp = FpVar::<Fr>::constant(Fr::from(l as u64));

        let n_total = n_valid.clone() * l_fp;

        let x_scaled = x_green.clone() * thousand.clone();
        let gamma_n = gamma_times_1000.clone() * n_total.clone();
        let delta_scaled = x_scaled - gamma_n;

        let one_minus_gamma = thousand.clone() - gamma_times_1000.clone();
        let gamma_one_minus_gamma = gamma_times_1000.clone() * one_minus_gamma;
        let v_scaled = n_total * gamma_one_minus_gamma;

        let delta_sq = delta_scaled.clone() * delta_scaled;
        let left = delta_sq * million;

        let tau_sq = tau_times_1000.clone() * tau_times_1000;
        let right = tau_sq * v_scaled;

        let slack = left - right;
        let _ = enforce_u_n(&slack, 192)?;

        // =========================
        // (3) 解码：✅ 修复为 g⊕b
        // =========================
        let max_cnt_u128: u128 = (n as u128) * (l as u128);
        let mut cnt_bits_cnt: usize = 1;
        while (1u128 << cnt_bits_cnt) <= (max_cnt_u128 + 1) {
            cnt_bits_cnt += 1;
        }

        let mut bit_vote_0: Vec<FpVar<Fr>> = vec![FpVar::<Fr>::zero(); bits_len];
        let mut bit_vote_1: Vec<FpVar<Fr>> = vec![FpVar::<Fr>::zero(); bits_len];

        for t in 0..n {
            let active_t = h_bits[t].clone();

            if FIX_BIT_IDX {
                let b_fixed = t % bits_len;
                for i in 0..l {
                    // ✅ 修复：改用 g_ti
                    let g_ti = g_matrix[t][i].clone();
                    let b_ti = b_matrix[t][i].clone();

                    // ✅ 修复：g⊕b
                    let m_hat_ti = Boolean::xor(&g_ti, &b_ti)?;

                    let add1 = Boolean::and(&active_t, &m_hat_ti)?;
                    let add0 = Boolean::and(&active_t, &m_hat_ti.not())?;

                    bit_vote_1[b_fixed] += FpVar::<Fr>::from(add1);
                    bit_vote_0[b_fixed] += FpVar::<Fr>::from(add0);
                }
            } else {
                let bit_idx_t = bit_idx_vars[t].clone();

                let mut eq: Vec<Boolean<Fr>> = Vec::with_capacity(bits_len);
                for b in 0..bits_len {
                    let bc = FpVar::<Fr>::constant(Fr::from(b as u64));
                    eq.push(bit_idx_t.is_eq(&bc)?);
                }
                let or_eq = or_reduce(&eq)?;

                let bad = Boolean::and(&active_t, &or_eq.not())?;
                bad.enforce_equal(&Boolean::constant(false))?;

                let mut should_vote: Vec<Boolean<Fr>> = Vec::with_capacity(bits_len);
                for b in 0..bits_len {
                    should_vote.push(Boolean::and(&active_t, &eq[b])?);
                }

                for i in 0..l {
                    // ✅ 修复：改用 g_ti
                    let g_ti = g_matrix[t][i].clone();
                    let b_ti = b_matrix[t][i].clone();

                    // ✅ 修复：g⊕b
                    let m_hat_ti = Boolean::xor(&g_ti, &b_ti)?;

                    for b in 0..bits_len {
                        let add1 = Boolean::and(&should_vote[b], &m_hat_ti)?;
                        let add0 = Boolean::and(&should_vote[b], &m_hat_ti.not())?;

                        bit_vote_1[b] += FpVar::<Fr>::from(add1);
                        bit_vote_0[b] += FpVar::<Fr>::from(add0);
                    }
                }
            }
        }

        for b in 0..bits_len {
            let v0 = bit_vote_0[b].clone();
            let v1 = bit_vote_1[b].clone();

            let v0_bits = enforce_u_n(&v0, cnt_bits_cnt)?;
            let v1_bits = enforce_u_n(&v1, cnt_bits_cnt)?;

            let m_rec_b = gt_bits_le(&v1_bits, &v0_bits)?;
            m_rec_b.enforce_equal(&m_bits[b])?;
        }

        // =========================
        // (4) BER
        // =========================
        let mut err_cnt = FpVar::<Fr>::zero();
        for b in 0..bits_len {
            let err_b = Boolean::xor(&m_bits[b], &orig_bits[b])?;
            err_cnt += FpVar::<Fr>::from(err_b);
        }

        let slack_err = e_max - err_cnt;
        let _ = enforce_u_n(&slack_err, 32)?;

        Ok(())
    }
}

// =========================
// I/O
// =========================
fn save_bin<T: CanonicalSerialize>(path: &str, obj: &T) -> Result<()> {
    let f = File::create(path).with_context(|| format!("create {}", path))?;
    let mut w = BufWriter::new(f);
    obj.serialize_uncompressed(&mut w)
        .with_context(|| format!("serialize {}", path))?;
    w.flush().ok();
    Ok(())
}

fn load_bin<T: CanonicalDeserialize>(path: &str) -> Result<T> {
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let mut r = BufReader::new(f);
    let obj = T::deserialize_uncompressed(&mut r).with_context(|| format!("deserialize {}", path))?;
    Ok(obj)
}

fn read_json<P: Into<PathBuf>>(p: P) -> Result<SnarkInput> {
    let p: PathBuf = p.into();
    let f = File::open(&p).with_context(|| format!("open json {:?}", p))?;
    let rdr = BufReader::new(f);
    let inp: SnarkInput = serde_json::from_reader(rdr).with_context(|| "parse json")?;
    inp.sanity_check()?;
    Ok(inp)
}

fn build_circuit(inp: SnarkInput) -> ProvMarkCircuit {
    ProvMarkCircuit { inp }
}

fn build_public_inputs(inp: &SnarkInput) -> Vec<Fr> {
    vec![
        Fr::from(inp.model_id_u64),
        Fr::from(inp.tau_times_1000),
        Fr::from(inp.gamma_times_1000),
        Fr::from(inp.e_max),
    ]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("ProvMark ZKP (GM17 + BLS12-381) - FIXED VERSION");
        println!("✅ Fix: Decoding formula changed from r⊕b to g⊕b");
        println!("");
        println!("Usage:");
        println!("  {} setup  <input.json> <pk.bin> <vk.bin>", args[0]);
        println!("  {} prove  <input.json> <pk.bin> <proof.bin>", args[0]);
        println!("  {} verify <input.json> <vk.bin> <proof.bin>", args[0]);
        bail!("missing arguments");
    }

    let cmd = args[1].as_str();

    match cmd {
        "setup" => {
            if args.len() < 5 {
                bail!("setup requires: <input.json> <pk.bin> <vk.bin>");
            }
            let input_path = &args[2];
            let pk_path = &args[3];
            let vk_path = &args[4];

            let inp = read_json(input_path)?;
            let circ = build_circuit(inp.clone());

            println!("🔧 GM17 setup...");
            println!("   n={}, L={}, bits_len={}", inp.n, inp.l, inp.bits_len);
            println!("   FIX_BIT_IDX = {}", FIX_BIT_IDX);

            let t = Instant::now();
            let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
            let (pk, vk) = GM17::<Bls12_381>::setup(circ, &mut rng).context("setup")?;
            println!("⏱️  setup: {} ms", t.elapsed().as_millis());

            save_bin(pk_path, &pk)?;
            save_bin(vk_path, &vk)?;
            println!("💾 pk: {}, vk: {}", pk_path, vk_path);
            println!("✅ setup done.");
        }

        "prove" => {
            if args.len() < 5 {
                bail!("prove requires: <input.json> <pk.bin> <proof.bin>");
            }
            let input_path = &args[2];
            let pk_path = &args[3];
            let proof_path = &args[4];

            let inp = read_json(input_path)?;
            let circ = build_circuit(inp.clone());

            let pk: ProvingKey<Bls12_381> = load_bin(pk_path)?;
            println!("🔧 GM17 prove...");
            println!("   n={}, L={}", inp.n, inp.l);

            let t = Instant::now();
            let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
            let proof = GM17::<Bls12_381>::prove(&pk, circ, &mut rng).context("prove")?;
            let elapsed = t.elapsed();
            println!("⏱️  prove: {} ms ({:.2}s)", elapsed.as_millis(), elapsed.as_secs_f64());

            save_bin(proof_path, &proof)?;
            println!("💾 proof: {}", proof_path);
            println!("✅ prove done.");
        }

        "verify" => {
            if args.len() < 5 {
                bail!("verify requires: <input.json> <vk.bin> <proof.bin>");
            }
            let input_path = &args[2];
            let vk_path = &args[3];
            let proof_path = &args[4];

            let inp = read_json(input_path)?;
            let public_inputs = build_public_inputs(&inp);

            let vk: VerifyingKey<Bls12_381> = load_bin(vk_path)?;
            let proof: Proof<Bls12_381> = load_bin(proof_path)?;

            println!("🔧 GM17 verify...");
            let t = Instant::now();
            let ok = GM17::<Bls12_381>::verify(&vk, &public_inputs, &proof).context("verify")?;
            println!("⏱️  verify: {} ms", t.elapsed().as_millis());

            if ok {
                println!("✅ verify: OK");
                println!("   model_id: {}", inp.model_id_u64);
                println!("   n={}, L={}, bits_len={}", inp.n, inp.l, inp.bits_len);
                println!("   active: {}/{}", inp.h.iter().filter(|&&x| x != 0).count(), inp.n);
                println!("   tau={:.3}, gamma={:.3}, e_max={}",
                    inp.tau_times_1000 as f64 / 1000.0,
                    inp.gamma_times_1000 as f64 / 1000.0,
                    inp.e_max);
                if let Some(db) = &inp.decoded_bits {
                    println!("   decoded: {}", db);
                }
            } else {
                println!("❌ verify: FAIL");
                println!("⚠️  可能原因:");
                println!("   1. pk/vk不匹配（用旧版本生成的）");
                println!("   2. JSON数据与proof不匹配");
                println!("   3. Python端用了错误的解码公式");
            }
        }

        _ => {
            bail!("Unknown command: {} (use setup/prove/verify)", cmd);
        }
    }

    Ok(())
}