/// Compile-time smoke test — verifies public API is accessible.
#[test]
fn public_api_compiles() {
    // Functions exist (don't need to call — just check they compile)
    let _ = ppsd_rs::cosine_taper as fn(usize, f64) -> Vec<f64>;
    let _ = ppsd_rs::fft_power as fn(&[f64], f64) -> (Vec<f64>, Vec<f64>);
    let _ = ppsd_rs::welch_psd as fn(&[f64], usize, f64, usize) -> (Vec<f64>, Vec<f64>);
    let _ = ppsd_rs::konno_ohmachi_smooth as fn(&[f64], &[f64], f64) -> Vec<f64>;

    // Response evaluation takes stationxml_rs::Response
    let _ =
        ppsd_rs::eval_response as fn(&stationxml_rs::Response, &[f64]) -> ppsd_rs::Result<Vec<f64>>;
}
