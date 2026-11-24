import os
import pandas as pd

SCHEDULES = [
    # District, granary_id, Cycle, Water_Start_Decade, Water_End_Decade, Stagger_Phases, Block_Stagger_Decades, PreSat_mm, Maint_mm, Regime
    ("MADA", "mada", "C1", 7, 12, 3, 1, 100, 100, "Dam_Fed"),
    ("MADA", "mada", "C2", 27, 33, 3, 1, 100, 100, "Dam_Fed"),
    ("KADA", "kada", "C1", 10, 15, 3, 1, 100, 100, "Run_of_River"),
    ("KADA", "kada", "C2", 28, 34, 3, 1, 100, 100, "Run_of_River"),
    ("IADA Barat Laut Selangor", "iada_barat_laut_selangor", "C1", 1, 12, 4, 3, 100, 100, "Run_of_River"),
    ("IADA Barat Laut Selangor", "iada_barat_laut_selangor", "C2", 19, 30, 4, 3, 100, 100, "Run_of_River"),
    ("IADA Kerian", "iada_kerian", "C1", 7, 12, 3, 1, 100, 100, "Dam_Fed"),
    ("IADA Kerian", "iada_kerian", "C2", 25, 30, 3, 1, 100, 100, "Dam_Fed"),
    ("KETARA", "iada_ketara", "C1", 13, 18, 2, 1, 100, 100, "Dam_Fed"),
    ("KETARA", "iada_ketara", "C2", 31, 36, 2, 1, 100, 100, "Dam_Fed"),
    ("IADA Pulau Pinang", "iada_pulau_pinang", "C1", 7, 12, 3, 1, 100, 100, "Run_of_River"),
    ("IADA Pulau Pinang", "iada_pulau_pinang", "C2", 25, 30, 3, 1, 100, 100, "Run_of_River"),
    ("IADA Seberang Perak", "iada_seberang_perak", "C1", 7, 12, 3, 1, 100, 100, "Run_of_River"),
    ("IADA Seberang Perak", "iada_seberang_perak", "C2", 25, 30, 3, 1, 100, 100, "Run_of_River"),
    ("IADA Kemasin Semerak", "iada_kemasin_semarak", "C1", 10, 15, 3, 1, 100, 100, "Run_of_River"),
    ("IADA Kemasin Semerak", "iada_kemasin_semarak", "C2", 28, 34, 3, 1, 100, 100, "Run_of_River"),
    ("IADA Pekan", "iada_pekan", "C1", 10, 15, 2, 1, 100, 100, "Run_of_River"),
    ("IADA Pekan", "iada_pekan", "C2", 28, 34, 2, 1, 100, 100, "Run_of_River"),
    ("IADA Rompin", "iada_rompin", "C1", 10, 15, 2, 1, 100, 100, "Run_of_River"),
    ("IADA Rompin", "iada_rompin", "C2", 28, 34, 2, 1, 100, 100, "Run_of_River"),
]

RULES = [
    ("R01_PLANT_SLIP", "Planting_Timing", "Pre_Sat", "Schedule_Slip_Decades", ">", 1.5, "Shift -1 to -2 decades within canal window", "Align seed soaking & wet direct seeding with official JPS/IADA Jadual Bekalan Air phase release.", "Paddy Calendar / MYSA MakGeoPadi"),
    ("R02_VEG_FLOOD", "Vegetative_Flooding", "Vegetative_0_40HST", "C1_MNDWI_veg", "<", -0.25, "Increase MNDWI_veg toward donor median", "Execute 1st pre-saturation (100 mm for 3 days) and build standing water to 50-100 mm post-germination for weed suppression.", "IADA Pulau Pinang Manual / FAO Ch.4"),
    ("R03_REP_LSWI", "Reproductive_Water", "Reproductive_40_70HST", "C1_LSWI_rep", "<", 0.25, "Raise LSWI_rep >= 0.28", "Prevent >10-day cyclic water deficit during panicle initiation & heading (40-70 HST); activate drainage recycling pumps if canal levels dip.", "UKM/UPM Cyclic Water Stress (2024)"),
    ("R04_STEM_BORER", "StemBorer_Scouting", "Vegetative_0_40HST", "StemBorer_Weather_Risk", "==", 1.0, "Deploy field scouting & pheromone traps", "Season-1 diurnal range >= 8.2C with <5 mm decade rain elevates yellow stem borer risk; scout for dead hearts before panicle initiation.", "MAB North Malaysia Stem Borer Study (2022)"),
]

def main():
    os.makedirs("external", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    cols_sched = [
        "District", "granary_id", "Cycle", "Water_Start_Decade", "Water_End_Decade",
        "Stagger_Phases", "Block_Stagger_Decades", "PreSaturation_Depth_mm",
        "Maintenance_Water_Depth_mm", "Irrigation_Regime"
    ]
    df_sched = pd.DataFrame(SCHEDULES, columns=cols_sched)
    assert df_sched["Water_Start_Decade"].between(1, 36).all(), "Invalid start decade"
    assert df_sched["Water_End_Decade"].between(1, 36).all(), "Invalid end decade"
    assert (df_sched["Water_Start_Decade"] <= df_sched["Water_End_Decade"]).all(), "Start > End decade"
    df_sched.to_csv("external/granary_irrigation_schedules.csv", index=False)

    cols_rules = [
        "Rule_ID", "Practice_Lever", "Growth_Stage", "Trigger_Column", "Operator",
        "Threshold_Value", "Target_Adjustment", "Field_Action_SOP", "Evidence_Citation"
    ]
    df_rules = pd.DataFrame(RULES, columns=cols_rules)
    assert not df_rules.isna().any().any(), "Null values in advisory_rules.csv"
    df_rules.to_csv("data/advisory_rules.csv", index=False)
    print("Verified and saved external/granary_irrigation_schedules.csv and data/advisory_rules.csv")

if __name__ == "__main__":
    main()