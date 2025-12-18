#!/usr/bin/env python3
"""
Python fastLowess validation runner with JSON output for comparison with statsmodels.

This validation program outputs results in JSON format compatible with the
Rust fastLowess validation results.
"""

import json
import os
from pathlib import Path

import numpy as np
import fastLowess


def main():
    # Same data as statsmodels validation
    x = np.array([
        0.0,
        0.06346651825433926,
        0.12693303650867852,
        0.1903995547630178,
        0.25386607301735703,
        0.3173325912716963,
        0.3807991095260356,
        0.4442656277803748,
        0.5077321460347141,
        0.5711986642890533,
        0.6346651825433925,
        0.6981317007977318,
        0.7615982190520711,
        0.8250647373064104,
        0.8885312555607496,
        0.9519977738150889,
        1.0154642920694281,
        1.0789308103237674,
        1.1423973285781066,
        1.2058638468324459,
        1.269330365086785,
        1.3327968833411243,
        1.3962634015954636,
        1.4597299198498028,
        1.5231964381041423,
        1.5866629563584815,
        1.6501294746128208,
        1.71359599286716,
        1.7770625111214993,
        1.8405290293758385,
        1.9039955476301778,
        1.967462065884517,
        2.0309285841388562,
        2.0943951023931957,
        2.1578616206475347,
        2.221328138901874,
        2.284794657156213,
        2.3482611754105527,
        2.4117276936648917,
        2.475194211919231,
        2.53866073017357,
        2.6021272484279097,
        2.6655937666822487,
        2.729060284936588,
        2.792526803190927,
        2.8559933214452666,
        2.9194598396996057,
        2.982926357953945,
        3.0463928762082846,
        3.1098593944626236,
        3.173325912716963,
        3.236792430971302,
        3.3002589492256416,
        3.3637254674799806,
        3.42719198573432,
        3.490658503988659,
        3.5541250222429985,
        3.6175915404973376,
        3.681058058751677,
        3.744524577006016,
        3.8079910952603555,
        3.8714576135146945,
        3.934924131769034,
        3.998390650023373,
        4.0618571682777125,
        4.1253236865320515,
        4.188790204786391,
        4.25225672304073,
        4.3157232412950695,
        4.3791897595494085,
        4.442656277803748,
        4.506122796058087,
        4.569589314312426,
        4.6330558325667655,
        4.696522350821105,
        4.759988869075444,
        4.823455387329783,
        4.886921905584122,
        4.950388423838462,
        5.013854942092801,
        5.07732146034714,
        5.14078797860148,
        5.204254496855819,
        5.267721015110158,
        5.331187533364497,
        5.394654051618837,
        5.458120569873176,
        5.521587088127515,
        5.585053606381854,
        5.648520124636194,
        5.711986642890533,
        5.775453161144872,
        5.838919679399211,
        5.902386197653551,
        5.96585271590789,
        6.029319234162229,
        6.092785752416569,
        6.156252270670908,
        6.219718788925247,
        6.283185307179586,
    ])
    
    y = np.array([
        0.24835707650561634,
        -0.005708230929027822,
        0.4504367226240955,
        -7.081010477562053,
        0.13407129981941124,
        0.19496496722389683,
        1.1612688634140234,
        -9.040305694591247,
        0.2514595431329927,
        0.8119208392485799,
        0.3611990826484093,
        0.4099227329014108,
        0.8110601472651291,
        -18.72007778531368,
        -3.9195455146237927,
        0.5334321874298493,
        0.3433098697823025,
        1.038577029745219,
        0.4556199575939129,
        0.2279960095974609,
        -9.251677464779888,
        0.8589234180802738,
        1.01857185535617,
        0.28146437135452573,
        0.7266759769204166,
        1.055335422528808,
        0.421357987240791,
        1.1776704510537686,
        0.6784831012553763,
        0.8179952836633037,
        7.174630623097578,
        1.8484933863590503,
        0.8892451619223689,
        0.3371699393064883,
        1.2438423106863659,
        0.18534001554532087,
        0.860181371856636,
        -0.2671408905610251,
        0.00267597606707648,
        0.7165896041551668,
        0.9362931538604761,
        12.285673966673729,
        0.40040238053329025,
        0.25037868761196924,
        -0.3972408518580448,
        -0.07818954735592465,
        -0.010008852693352943,
        0.6865625090828078,
        0.21829574157811404,
        -0.8497921441832993,
        0.13031405119932965,
        -0.28759718351234087,
        -0.4964623961263295,
        0.08552761163389339,
        0.23376720440654564,
        0.12361991623243063,
        10.206335357609184,
        -0.6128327096530176,
        -0.34804567587162444,
        -0.07928730030159103,
        -0.8577461051432502,
        -0.7595984888482,
        -1.2658616583818771,
        -1.3538528863945936,
        -0.3894989293337331,
        -0.15444984034935982,
        -0.9020304645746058,
        -0.3942273253453239,
        -0.7415362815807642,
        -1.2675606960172305,
        -0.7831443558057352,
        -0.20978416298179414,
        -1.0077344614359085,
        -0.21453294804493916,
        -2.309746679718747,
        -0.5879160869953961,
        -0.9503149303421685,
        -1.1343114282451419,
        -0.9259311800557906,
        -1.9486866987445204,
        -1.0439838041838627,
        -0.731075709598645,
        -0.14250634107682392,
        -1.1088605390863382,
        -1.2188227534969296,
        -1.0270249860840248,
        -0.27689064980649614,
        -0.5257034566522698,
        -0.907667711570059,
        -0.33627421249796213,
        -0.49210204278157726,
        -0.00187424083402421,
        -0.7808214590278482,
        -0.5354935289592113,
        -0.5080875222645659,
        -29.7774743959709,
        -0.041191105828121716,
        0.003935182516195623,
        -0.060867191335334074,
        -0.11729356668757371,
    ])

    results = {}

    # Scenario 1: basic (default parameters)
    result = fastLowess.smooth(x, y)
    results["basic"] = format_result(result)

    # Scenario 2: small_fraction
    result = fastLowess.smooth(x, y, fraction=0.2)
    results["small_fraction"] = format_result(result)

    # Scenario 3: no_robust (iterations=0)
    result = fastLowess.smooth(x, y, iterations=0)
    results["no_robust"] = format_result(result)

    # Scenario 4: more_robust (iterations=5)
    result = fastLowess.smooth(x, y, iterations=5)
    results["more_robust"] = format_result(result)

    # Scenario 5: auto_converge
    result = fastLowess.smooth(x, y, auto_converge=1e-4)
    results["auto_converge"] = format_result(result)

    # Scenario 6: cross_validate (k-fold)
    result = fastLowess.smooth(
        x, y,
        cv_fractions=[0.2, 0.4, 0.6],
        cv_method="kfold",
        cv_k=5,
    )
    results["cross_validate"] = format_result(result)

    # Scenario 7: kfold_cv (same as cross_validate)
    result = fastLowess.smooth(
        x, y,
        cv_fractions=[0.2, 0.4, 0.6],
        cv_method="kfold",
        cv_k=5,
    )
    results["kfold_cv"] = format_result(result)

    # Scenario 8: loocv
    result = fastLowess.smooth(
        x, y,
        cv_fractions=[0.2, 0.4, 0.6],
        cv_method="loocv",
    )
    results["loocv"] = format_result(result)

    # Scenario 9: delta_zero
    result = fastLowess.smooth(x, y, delta=0.0)
    results["delta_zero"] = format_result(result)

    # Scenario 10: with_all_diagnostics
    result = fastLowess.smooth(
        x, y,
        return_diagnostics=True,
        return_residuals=True,
        return_robustness_weights=True,
    )
    results["with_all_diagnostics"] = format_result(result)

    # Output to JSON
    json_str = json.dumps(results, indent=2)

    # Determine output directory (parent's output dir: ../output)
    script_dir = Path(__file__).parent
    out_dir = script_dir.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "fastLowess_validate.json"
    with open(out_path, "w") as f:
        f.write(json_str)

    print(f"Saved results to {out_path}")


def format_result(result):
    """Format a LowessResult into a dictionary for JSON serialization."""
    scenario_data = {
        "x": result.x.tolist(),
        "y": result.y.tolist(),
        "residuals": result.residuals.tolist() if result.residuals is not None else None,
        "robustness_weights": result.robustness_weights.tolist() if result.robustness_weights is not None else None,
        "diagnostics": None,
        "iterations_used": result.iterations_used,
        "fraction_used": result.fraction_used,
        "cv_scores": result.cv_scores.tolist() if result.cv_scores is not None else None,
    }

    if result.diagnostics is not None:
        scenario_data["diagnostics"] = {
            "rmse": result.diagnostics.rmse,
            "mae": result.diagnostics.mae,
            "r_squared": result.diagnostics.r_squared,
            "aic": result.diagnostics.aic,
            "aicc": result.diagnostics.aicc,
            "effective_df": result.diagnostics.effective_df,
            "residual_sd": result.diagnostics.residual_sd,
        }

    return scenario_data


if __name__ == "__main__":
    main()
