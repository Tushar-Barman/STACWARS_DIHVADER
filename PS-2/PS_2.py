import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_raw():  # Loading the raw dataset
    df = pd.read_csv("SolarPrediction.csv")
    print("ROWS AND COLUMNS LOADED")
    print(df.shape)
    return df

# Extracting Date and appending Time to the DateTime column
def add_datetime(df):
    df = df.copy()
    date = df["Data"].astype(str).str.split().str[0]

    df["Datetime"] = pd.to_datetime(
        date + " " + df["Time"].astype(str).str.strip(),
        format="%m/%d/%Y %H:%M:%S",
        errors="coerce"
    )
    count = df["Datetime"].isna().sum()
    print("Number of invalid date/time values:", count)

    if count > 0:
        print("Few invalid date/time rows:")
        print(df[df["Datetime"].isna()].head(5))
        df = df.dropna(subset=["Datetime"]).copy()

    return df
def add_sunrise_sunset(df):
    df = df.copy()
    date = df["Datetime"].dt.strftime("%m/%d/%Y")
    df["sunrise"] = pd.to_datetime(date + " " + df["TimeSunRise"].astype(str).str.strip(),format="%m/%d/%Y %H:%M:%S",errors="coerce")
    df["sunset"] = pd.to_datetime(date + " " + df["TimeSunSet"].astype(str).str.strip(),format="%m/%d/%Y %H:%M:%S",errors="coerce")
    count = (df["sunrise"].isna() | df["sunset"].isna() | (df["sunrise"] >= df["sunset"])).sum()

    print("Rows with missing or invalid sunrise/sunset:", count)
    return df


def clean(df):
    print("Row Count before", len(df))

    df = df.sort_values(by="Datetime")

    print("Number Of Duplicated Rows:", df.duplicated().sum())
    df.drop_duplicates(inplace=True, ignore_index=True)

    print(df["Datetime"].diff().value_counts().head())
    print("Row Count after", len(df))

    return df


def save_clean(df):
    df.to_csv("SolarPrediction_processed.csv", index=False)
    print("Cleaned rows saved:", len(df))

#PART A: DATA VISUALISATION

def plot_radiation_time(df):
    plt.figure(figsize=(12, 5))
    plt.plot(df["Datetime"], df["Radiation"], linewidth=0.8)
    plt.title("Solar Radiation Over Time")
    plt.xlabel("Date and Time")
    plt.ylabel("Radiation (dataset units)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/radiation_over_time.png", dpi=160)
    plt.close()

# Daily mean radiation
    daily_mean = (
        df.dropna(subset=["Datetime", "Radiation"]).assign(date=lambda x: x["Datetime"].dt.date).groupby("date")["Radiation"].mean())
    plt.figure(figsize=(12, 5))
    plt.plot(daily_mean.index,daily_mean.values,marker="o",linewidth=1)
    plt.title("Daily Mean Solar Radiation Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mean radiation (dataset units)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("figures/daily_mean_radiation.png", dpi=160)
    plt.close()

    # Detailed view of the first three days
    valid = (
        df.dropna(subset=["Datetime", "Radiation"]).sort_values("Datetime"))
    if not valid.empty:
        start_time = valid["Datetime"].min().normalize()
        end_time = start_time + pd.Timedelta(days=3)

        zoom = valid[(valid["Datetime"] >= start_time)& (valid["Datetime"] < end_time)]
        if not zoom.empty:
            plt.figure(figsize=(12, 5))
            plt.plot(zoom["Datetime"],zoom["Radiation"],linewidth=0.9)
            plt.title("Solar Radiation: Three-Day Detailed View")
            plt.xlabel("Date and Time")
            plt.ylabel("Radiation (dataset units)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                "figures/radiation_three_day_zoom.png",
                dpi=160
            )
            plt.close()
def plot_radiation_relationships(df):
    variables = ["Temperature", "Humidity", "Pressure", "Speed"]

    for col in variables:
        if col in df.columns:
            temp = df[[col, "Radiation"]].apply(pd.to_numeric,errors="coerce").dropna()
            plt.figure(figsize=(7, 5))
            plt.scatter(temp[col],temp["Radiation"],s=10,alpha=0.4)
            plt.title("Radiation vs " + col)
            plt.xlabel(col)
            plt.ylabel("Radiation (dataset units)")
            plt.tight_layout()
            plt.savefig("figures/radiation_vs_" + col.lower() + ".png",dpi=160)
            plt.close()


def plot_radiation_distribution(df):
    values = pd.to_numeric(df["Radiation"],errors="coerce").dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40, edgecolor="black")
    plt.title("Distribution of Solar Radiation")
    plt.xlabel("Radiation (dataset units)")
    plt.ylabel("Number of observations")
    plt.tight_layout()
    plt.savefig("figures/radiation_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.boxplot(values, vert=False)
    plt.title("Solar Radiation Box Plot")
    plt.xlabel("Radiation (dataset units)")
    plt.tight_layout()
    plt.savefig("figures/radiation_boxplot.png", dpi=160)
    plt.close()

# PART B: DATA ANALYSIS AND INFERENCES

def analyse_correlations(df):
    variables = ["Temperature", "Humidity", "Pressure", "Speed"]
    rows = []
    for col in variables:
        if col in df.columns:
            pair = df[[col, "Radiation"]].apply(pd.to_numeric,errors="coerce").dropna()
            corr = (pair["Radiation"].corr(pair[col]) if len(pair) > 1 and pair[col].nunique() > 1 else np.nan)
            rows.append({"Parameter": col,"Pearson correlation with radiation": corr,"Absolute correlation": (abs(corr) if pd.notna(corr) else np.nan),"Observations": len(pair)})

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values("Absolute correlation",ascending=False)
    result.to_csv("outputs/radiation_correlations.csv",index=False)
    print("\nCORRELATION SUMMARY:")
    print(result.to_string(index=False))
    valid = result.dropna(subset=["Pearson correlation with radiation"])
    if not valid.empty:
        strongest = valid.loc[valid["Absolute correlation"].idxmax()]

        print("Strongest linear relationship with radiation:",strongest["Parameter"],
            f"(Pearson r = "f"{strongest['Pearson correlation with radiation']:.3f})")

        for _, row in valid.iterrows():
            r = row["Pearson correlation with radiation"]

            if abs(r) >= 0.7:
                strength = "strong"
            elif abs(r) >= 0.4:
                strength = "moderate"
            elif abs(r) >= 0.2:
                strength = "weak"
            else:
                strength = "very weak"

            if r > 0:
                direction = "positive"
            elif r < 0:
                direction = "negative"
            else:
                direction = "no"

            print(f"{row['Parameter']}: {strength} {direction} "f"linear association (r = {r:.3f}).")

    return result

def analyse_daily_patterns(df):
    work = df.dropna(subset=["Datetime", "Radiation"]).copy()

    work["Radiation"] = pd.to_numeric(work["Radiation"],errors="coerce")
    work = work.dropna(subset=["Radiation"])
    work["hour"] = work["Datetime"].dt.hour
    hourly = (
        work.groupby("hour")["Radiation"].agg(mean_radiation="mean",median_radiation="median",readings="count").reset_index())
    hourly.to_csv("outputs/hourly_radiation_summary.csv",index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(hourly["hour"],hourly["mean_radiation"],marker="o",label="Mean")
    plt.plot(hourly["hour"],hourly["median_radiation"],marker=".",label="Median")
    plt.title("Solar Radiation by Hour of Day")
    plt.xlabel("Hour of Day")
    plt.ylabel("Radiation (dataset units)")
    plt.xticks(range(0, 24, 2))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/radiation_by_hour.png", dpi=160)
    plt.close()

    print("\nHOURLY RADIATION SUMMARY:")
    print(hourly.to_string(index=False))

    if not hourly.empty:
        peak_hour = hourly.loc[hourly["mean_radiation"].idxmax()]

        hour_value = int(peak_hour["hour"])
        print("Solar radiation is generally highest from "
            f"{hour_value:02d}:00 to {hour_value + 1:02d}:00 "
            f"(hourly mean = {peak_hour['mean_radiation']:.2f}; "
            f"median = {peak_hour['median_radiation']:.2f}; "
            f"readings = {int(peak_hour['readings'])})."
        )
        daytime = hourly[hourly["hour"].between(6, 18)]
        outside_daytime = hourly[~hourly["hour"].between(6, 18)]
        if not daytime.empty:
            print("Average of hourly mean radiation for 06:00-18:00:",f"{daytime['mean_radiation'].mean():.2f}")
        if not outside_daytime.empty:
            print("Average of hourly mean radiation outside 06:00-18:00:",f"{outside_daytime['mean_radiation'].mean():.2f}")
    if "sunrise" in work.columns and "sunset" in work.columns:
        daylight_work = work.dropna(subset=["sunrise", "sunset"]).copy()
        daylight_work["period"] = np.where((daylight_work["Datetime"] >= daylight_work["sunrise"])& (daylight_work["Datetime"] <= daylight_work["sunset"]),"Daylight","Outside daylight")
        daylight_summary = (daylight_work.groupby("period")["Radiation"].agg(mean_radiation="mean",median_radiation="median",readings="count").reset_index())
        daylight_summary.to_csv("outputs/daylight_radiation_summary.csv",index=False)

        print("RADIATION DURING DAYLIGHT AND OUTSIDE DAYLIGHT:")
        print(daylight_summary.to_string(index=False))

        plt.figure(figsize=(8, 5))
        plt.bar(daylight_summary["period"],daylight_summary["mean_radiation"])
        plt.title("Mean Radiation During and Outside Daylight")
        plt.xlabel("Period")
        plt.ylabel("Mean radiation (dataset units)")
        plt.tight_layout()
        plt.savefig("figures/daylight_radiation_comparison.png",dpi=160)
        plt.close()
    return hourly
def analyse_temperature_humidity_patterns(df):
    work = df.copy()

    for col in ["Radiation", "Temperature", "Humidity"]:
        work[col] = pd.to_numeric(work[col],errors="coerce")
    work = work.dropna(subset=["Radiation", "Temperature", "Humidity"])
    if work.empty:
        print("Temperature-humidity analysis skipped: no complete numeric records.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    correlations = work[["Radiation", "Temperature", "Humidity"]].corr()

    correlations.to_csv("outputs/radiation_temperature_humidity_correlations.csv")
    work["Humidity_Group"] = pd.cut(work["Humidity"],bins=[-np.inf, 20, 40, 60, 80, np.inf],labels=["<=20", "20-40", "40-60", "60-80", ">80"])

    humidity_summary = (work.groupby("Humidity_Group", observed=False).agg(mean_radiation=("Radiation", "mean"),median_radiation=("Radiation", "median"),mean_temperature=("Temperature", "mean"),readings=("Radiation", "count")).reset_index())
    humidity_summary.to_csv("outputs/radiation_by_humidity_group.csv",index=False)

    # Examine temperature-radiation relationships within humidity groups.
    group_rows = []

    for group, subset in work.groupby("Humidity_Group",observed=False):
        subset = subset.dropna(subset=["Radiation", "Temperature"])
        r = (subset["Radiation"].corr(subset["Temperature"])if len(subset) > 1and subset["Radiation"].nunique() > 1 and subset["Temperature"].nunique() > 1
            else np.nan
        )
        group_rows.append({"Humidity_Group": str(group),"Observations": len(subset),"Mean_Radiation": subset["Radiation"].mean(),"Mean_Temperature": subset["Temperature"].mean(),"Temperature_Radiation_Correlation": r
        })

    grouped_relationships = pd.DataFrame(group_rows)
    grouped_relationships.to_csv("outputs/temperature_radiation_by_humidity_group.csv",index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(work["Temperature"],work["Radiation"],c=work["Humidity"],cmap="viridis",s=12,alpha=0.55)
    plt.colorbar(label="Humidity")
    plt.title("Radiation vs Temperature (colour indicates Humidity)")
    plt.xlabel("Temperature")
    plt.ylabel("Radiation (dataset units)")
    plt.tight_layout()
    plt.savefig("figures/radiation_temperature_humidity.png",dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(humidity_summary["Humidity_Group"].astype(str),humidity_summary["mean_radiation"])
    plt.title("Mean Radiation by Humidity Group")
    plt.xlabel("Humidity group")
    plt.ylabel("Mean radiation (dataset units)")
    plt.tight_layout()
    plt.savefig("figures/radiation_by_humidity_group.png",dpi=160
    )
    plt.close()

    print("\nRADIATION, TEMPERATURE AND HUMIDITY PATTERNS:")
    print("Correlation matrix:")
    print(correlations.round(3).to_string())

    print("Radiation grouped by humidity:")
    print(humidity_summary.to_string(index=False))

    print("Temperature-radiation relationship within humidity groups:")
    print(grouped_relationships.to_string(index=False))

    r_temp = correlations.loc["Radiation", "Temperature"]
    r_humidity = correlations.loc["Radiation", "Humidity"]

    print(f"\nOverall radiation-temperature correlation: {r_temp:.3f}; "f"radiation-humidity correlation: {r_humidity:.3f}.")

    if pd.notna(r_temp):
        if r_temp > 0:
            print("Overall, higher temperature tends to accompany higher radiation.")
        elif r_temp < 0:
            print("Overall, higher temperature tends to accompany lower radiation.")
        else:
            print("The overall linear temperature-radiation association is close to zero.")

    if pd.notna(r_humidity):
        if r_humidity > 0:
            print("Overall, higher humidity tends to accompany higher radiation.")
        elif r_humidity < 0:
            print(
                "Overall, higher humidity tends to accompany lower radiation.")
        else:
            print("The overall linear humidity-radiation association is close to zero.")

    print("The grouped correlations show whether the temperature-radiation relationship changes across humidity ranges. These are associations, not proof of causation.")

    return correlations, humidity_summary, grouped_relationships


def analyse_outliers(df):
    work = df.copy()
    work["Radiation_numeric"] = pd.to_numeric(work["Radiation"], errors="coerce")
    values = work["Radiation_numeric"].dropna()
    if values.empty:
        print("No valid radiation values available for outlier analysis.")
        return work.iloc[0:0].copy()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = work[(work["Radiation_numeric"] < lower)|(work["Radiation_numeric"] > upper) ].copy()
    outliers["outlier_type"] = np.where(outliers["Radiation_numeric"] > upper,"High","Low" )

    if all(
        col in outliers.columns
        for col in ["Datetime", "sunrise", "sunset"]):
        outliers["outside_daylight"] = ((outliers["Datetime"] < outliers["sunrise"])| (outliers["Datetime"] > outliers["sunset"]))
    else:
        outliers["outside_daylight"] = np.nan

    # Save outliers with environmental context.
    context_columns = [c for c in ["Datetime","Radiation","outlier_type","outside_daylight","Temperature","Humidity","Pressure","Speed","sunrise","sunset"]if c in outliers.columns]
    outliers[context_columns].to_csv("outputs/radiation_outliers.csv",index=False)
    high = outliers[outliers["outlier_type"] == "High"]
    low = outliers[outliers["outlier_type"] == "Low"]
    print("OUTLIER ANALYSIS:")
    print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"Lower threshold: {lower:.2f}")
    print(f"Upper threshold: {upper:.2f}")

    print(f"Potential outliers: {len(outliers)} "f"({len(high)} high, {len(low)} low)")
    if (not outliers.empty and "outside_daylight" in outliers.columns):
        if outliers["outside_daylight"].notna().any():
            outside_count = int(outliers["outside_daylight"].fillna(False).sum())
            print("Outliers outside the recorded sunrise-sunset interval:",outside_count)

    if not high.empty:
        print("Highest radiation outliers (first 10):")
        print(high.sort_values("Radiation_numeric",ascending=False)[context_columns].head(10).to_string(index=False))
    if not low.empty:
        print("Lowest radiation outliers (first 10):")
        print(low.sort_values("Radiation_numeric",ascending=True)[context_columns].head(10).to_string(index=False))
    # Summarise environmental conditions for high and low outliers.
    environment = [c for c in ["Temperature","Humidity","Pressure","Speed"]if c in outliers.columns]
    if environment and not outliers.empty:
        outlier_environment = (outliers.groupby("outlier_type")[environment].agg(["mean", "median", "min", "max"]))
        outlier_environment.to_csv("outputs/outlier_environment_summary.csv")
        print("Environmental summary for high and low outliers:")
        print(outlier_environment.to_string())
    print("Interpretation: IQR identifies unusually high or low radiation, "
        "but it does not establish the cause. Use the saved timestamps and "
        "weather readings to investigate possible cloud variation, unusual "
        "conditions, or measurement/data-quality issues.")

    return outliers


def member2_analysis(df):
    print("PART A: VISUALISATION")

    os.makedirs("figures", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    plot_radiation_time(df)
    plot_radiation_relationships(df)
    plot_radiation_distribution(df)
    print("PART B: INFERENCES")
    correlations = analyse_correlations(df)
    hourly = analyse_daily_patterns(df)(
        temp_humidity_correlations,
        humidity_summary,
        grouped_relationships
    ) = analyse_temperature_humidity_patterns(df)
    outliers = analyse_outliers(df)
    print("PART B SUMMARY:")
    print("The highest-radiation hour is identified from hourly mean radiation.")
    print("Pearson correlations rank the environmental parameters by linear association.")
    print("Temperature and humidity are examined overall and acrosshumidity groups.")
    print("High and low IQR outliers are saved with their timestamps and available weather readings.")
    print("-Correlation does not establish causation, and an outlier is not automatically an error.")

    return correlations, hourly, outliers

#  PART C: DAILY REPORT
def fmt(td):
    s = int(td.total_seconds())
    return f"{s // 3600} h {(s % 3600) // 60} min"

def daily_summary(df):
    daily = (df.groupby("date")["Radiation"].agg(max_radiation="max",mean_radiation="mean",n_readings="count") .reset_index())
    assert len(daily) == df["date"].nunique()
    daily.to_csv("outputs/daily_summary.csv", index=False)
    print("\nDAILY SUMMARY:\n", daily)
    return daily

def top_radiation_daily(daily, df):
    top = daily["max_radiation"].max()
    assert top == df["Radiation"].max()
    top_dates = daily.loc[daily["max_radiation"] == top,"date"].tolist()
    print("HIGHEST RADIATION:", top, "on", top_dates)
    return top_dates
def peak_conditions(df, top_dates):
    peaks = []
    for d in top_dates:
        that_day = df[df["date"] == d]
        peaks.append(that_day[that_day["Radiation"] == that_day["Radiation"].max()])
    peak = pd.concat(peaks)

    print("CONDITIONS AT PEAK:")

    columns = [c for c in ["Datetime","Radiation","Temperature","Humidity", "Speed"]if c in peak.columns]
    print(peak[columns].to_string(index=False))
    peak[columns].to_csv("outputs/peak_conditions.csv",index=False)

    return peak


def peak_vs_sun(peak):
    results = []
    for _, row in peak.iterrows():
        if pd.isna(row["sunrise"]) or pd.isna(row["sunset"]):
            after = before = length = pd.NaT
        else:
            after = row["Datetime"] - row["sunrise"]
            before = row["sunset"] - row["Datetime"]
            length = row["sunset"] - row["sunrise"]
        print(
            f"\nPeak at {row['Datetime'].time()}: "
            f"{fmt(after) if pd.notna(after) else 'unavailable'} "
            f"after sunrise "
            f"({row['sunrise'].time() if pd.notna(row['sunrise']) else 'unavailable'}), "
            f"{fmt(before) if pd.notna(before) else 'unavailable'} "
            f"before sunset "
            f"({row['sunset'].time() if pd.notna(row['sunset']) else 'unavailable'}); "
            f"day length "
            f"{fmt(length) if pd.notna(length) else 'unavailable'}"
        )
        results.append({"peak_time": row["Datetime"],"after_sunrise": (fmt(after) if pd.notna(after) else "unavailable"),"before_sunset": (fmt(before) if pd.notna(before) else "unavailable"),"day_length": (fmt(length) if pd.notna(length) else "unavailable")})
    return pd.DataFrame(results)
def top5_by_mean(daily):
    top5 = daily.nlargest(5,"mean_radiation")[["date","mean_radiation","max_radiation"]]
    assert len(top5) == 5
    print("TOP 5 DAYS BY MEAN RADIATION:\n",top5.to_string(index=False))

    return top5
# MAIN FUNCTION

def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    df = load_raw()
    df = add_datetime(df)
    df = add_sunrise_sunset(df)
    df = clean(df)
    save_clean(df)

    df["Radiation"] = pd.to_numeric(df["Radiation"],errors="coerce")
    df["date"] = df["Datetime"].dt.date
    # Member 2: Parts A and B
    member2_analysis(df)
    # Member 1: Part C and Data Cleaning
    daily = daily_summary(df)
    top_dates = top_radiation_daily(daily, df)
    peak = peak_conditions(df, top_dates)
    timing = peak_vs_sun(peak)
    timing.to_csv("outputs/peak_vs_sun.csv",index=False)
    top5 = top5_by_mean(daily)
    top5.to_csv("outputs/top5_by_mean.csv",index=False)
if __name__ == "__main__":
    main()