import numpy as np
import pandas as pd


def build_profiles_and_adaptive_price_baseline(
    df,
    date_col="date",
    price_col="price",
    label_weight_col="label_weight",

    # فقط برای حذف قیمت‌های نامعتبر
    min_price=1.0,

    min_monthly_support=7,
    min_weekly_support=3,

    # momentum نسبی
    momentum_clip=0.20,
):
    """
    Past-only price profile engine.

    این نسخه به جای target_ratio مستقیماً از PRICE استفاده می‌کند.

    ساختار:
    - L1 / L2 / L3 / L4
    - Daily / Weekly / Monthly
    - Time weighted profiles
    - Monthly: time_weight * label_weight
    - Weighted median / MAD / P10 / P25 / P75 / P90
    - IQR / STD / CV / skewness / sample_count
    - Hierarchical monthly price baseline
    - Hierarchical weekly price baseline
    - Parent fallback
    - Adaptive price baseline
    - Relative momentum
    - Baseline confidence
    - Profile support
    - Price residual
    - Log price residual

    نکته مهم:
    تمام profileها strictly past-only هستند.
    """

    # =========================================================
    # 1. PREPARATION
    # =========================================================

    df = df.copy()

    prefixes_to_drop = [
        "price_l",
        "parent_",
        "baseline",
        "adaptive_",
        "momentum_",
        "profile_support",
    ]

    old_cols = [
        c
        for c in df.columns
        if any(c.startswith(p) for p in prefixes_to_drop) and c != 'parent_id'
    ]

    if old_cols:
        df = df.drop(columns=old_cols)

    original_len = len(df)

    # ---------------------------------------------------------
    # Date
    # ---------------------------------------------------------

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Price
    # ---------------------------------------------------------

    df[price_col] = pd.to_numeric(
        df[price_col],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Label weight
    # ---------------------------------------------------------

    df[label_weight_col] = pd.to_numeric(
        df[label_weight_col],
        errors="coerce"
    ).fillna(1.0)

    df[label_weight_col] = (
        df[label_weight_col]
        .clip(lower=1e-6)
        .astype("float32")
    )

    # ---------------------------------------------------------
    # Internal row id
    # ---------------------------------------------------------

    df["_profile_row_id"] = np.arange(
        len(df),
        dtype=np.int64
    )

    # ---------------------------------------------------------
    # Sort by date
    # ---------------------------------------------------------

    df = (
        df
        .sort_values(
            date_col,
            kind="mergesort"
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # 2. GROUP DEFINITIONS
    # =========================================================

    group_defs = {

        "l1": [
            "real_brand",
            "real_model",
            "is_copy",
            "model_variant",
            "age",
        ],

        "l2": [
            "real_brand",
            "real_model",
            "is_copy",
            "age",
        ],
        "l3":[
            "real_model",
            "is_copy",
            "age",
        ],

        "l4": [
            "real_brand",
            "segment",
            "is_copy",
            "age",
        ],

        "l5": [
            "segment",
            "is_copy",
            "age",
        ],
    }

    required_group_cols = set()

    for cols in group_defs.values():
        required_group_cols.update(cols)

    missing_group_cols = [
        c
        for c in required_group_cols
        if c not in df.columns
    ]

    if missing_group_cols:
        raise ValueError(
            f"Missing group columns: {missing_group_cols}"
        )

    # =========================================================
    # 3. PROFILE SOURCE
    # =========================================================
    #
    # در نسخه قبلی:
    # target_ratio بین 0.4 و 1.6 شرط داشت.
    #
    # در نسخه قیمتی چنین شرطی نداریم.
    # فقط:
    # - price معتبر
    # - date معتبر
    # - price > min_price
    #
    # =========================================================

    profile_source = df.loc[
        df[price_col].notna()
        &
        (df[price_col] >= min_price)
        &
        df[date_col].notna()
    ].copy()

    profile_source = (
        profile_source
        .sort_values(
            date_col,
            kind="mergesort"
        )
        .reset_index(drop=True)
    )

    print("=" * 70)
    print("PRICE PROFILE SOURCE")
    print("=" * 70)

    print(
        f"Total rows       : {len(df):,}"
    )

    print(
        f"Profile rows     : {len(profile_source):,}"
    )

    print(
        f"Profile excluded : "
        f"{len(df) - len(profile_source):,}"
    )

    # =========================================================
    # 4. WEIGHTED QUANTILE
    # =========================================================

    def weighted_quantile(
        values,
        weights,
        q,
    ):

        values = np.asarray(
            values,
            dtype=np.float64
        )

        weights = np.asarray(
            weights,
            dtype=np.float64
        )

        valid = (
            np.isfinite(values)
            &
            np.isfinite(weights)
            &
            (weights > 0)
        )

        values = values[valid]
        weights = weights[valid]

        if values.size == 0:
            return np.nan

        order = np.argsort(
            values,
            kind="mergesort"
        )

        values = values[order]
        weights = weights[order]

        cumulative = np.cumsum(
            weights
        )

        threshold = (
            q
            * cumulative[-1]
        )

        idx = np.searchsorted(
            cumulative,
            threshold,
            side="left"
        )

        idx = min(
            idx,
            len(values) - 1
        )

        return float(
            values[idx]
        )

    # =========================================================
    # 5. WEIGHTED PROFILE STATISTICS
    # =========================================================

    def weighted_stats(
        values,
        weights,
    ):

        values = np.asarray(
            values,
            dtype=np.float64
        )

        weights = np.asarray(
            weights,
            dtype=np.float64
        )

        valid = (
            np.isfinite(values)
            &
            np.isfinite(weights)
            &
            (weights > 0)
        )

        values = values[valid]
        weights = weights[valid]

        n = len(values)

        if n == 0:
            return {
                "median": np.nan,
                "mad": np.nan,
                "p10": np.nan,
                "p90": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "iqr": np.nan,
                "std": np.nan,
                "cv": np.nan,
                "skewness": np.nan,
                "sample_count": 0,
            }

        median = weighted_quantile(
            values,
            weights,
            0.50
        )

        p10 = weighted_quantile(
            values,
            weights,
            0.10
        )

        p25 = weighted_quantile(
            values,
            weights,
            0.25
        )

        p75 = weighted_quantile(
            values,
            weights,
            0.75
        )

        p90 = weighted_quantile(
            values,
            weights,
            0.90
        )

        iqr = p75 - p25

        weight_sum = weights.sum()

        mean = (
            np.sum(
                values * weights
            )
            / weight_sum
        )

        variance = (
            np.sum(
                weights
                * (values - mean) ** 2
            )
            / weight_sum
        )

        std = np.sqrt(
            max(
                variance,
                0.0
            )
        )

        abs_deviation = np.abs(
            values - median
        )

        mad = weighted_quantile(
            abs_deviation,
            weights,
            0.50
        )

        if abs(mean) > 1e-12:
            cv = (
                std
                / abs(mean)
            )
        else:
            cv = np.nan

        if std > 1e-12:

            third_moment = (
                np.sum(
                    weights
                    * (values - mean) ** 3
                )
                / weight_sum
            )

            skewness = (
                third_moment
                / (std ** 3)
            )

        else:
            skewness = 0.0

        return {
            "median": median,
            "mad": mad,
            "p10": p10,
            "p90": p90,
            "p25": p25,
            "p75": p75,
            "iqr": iqr,
            "std": std,
            "cv": cv,
            "skewness": skewness,
            "sample_count": n,
        }

    # =========================================================
    # 6. BUILD PROFILE TABLE
    # =========================================================

    def build_profile_table(
        source,
        level_name,
        group_cols,
    ):

        print(
            f"\nBuilding {level_name} PRICE profiles..."
        )

        source = source[
            group_cols
            + [
                date_col,
                price_col,
                label_weight_col,
            ]
        ].copy()

        source = (
            source
            .sort_values(
                date_col,
                kind="mergesort"
            )
            .reset_index(drop=True)
        )

        # -----------------------------------------------------
        # Sorted datetime array
        # -----------------------------------------------------

        date_array = (
            source[date_col]
            .values
            .astype("datetime64[ns]")
        )

        unique_dates = np.unique(
            date_array
        )

        output_rows = []

        total_dates = len(unique_dates)

        for date_idx, current_date64 in enumerate(
            unique_dates,
            start=1
        ):

            current_date = pd.Timestamp(
                current_date64
            )

            # -------------------------------------------------
            # STRICTLY PAST
            # -------------------------------------------------

            right_idx = np.searchsorted(
                date_array,
                current_date64,
                side="left"
            )

            if right_idx == 0:
                continue

            # -------------------------------------------------
            # WINDOWS
            # -------------------------------------------------

            for window_name, window_days in (
                ("daily", 1),
                ("weekly", 7),
                ("monthly", 30),
            ):

                window_start64 = (
                    current_date64
                    -
                    np.timedelta64(
                        window_days,
                        "D"
                    )
                )

                left_idx = np.searchsorted(
                    date_array,
                    window_start64,
                    side="left"
                )

                if left_idx >= right_idx:
                    continue

                hist = source.iloc[
                    left_idx:right_idx
                ].copy()

                if hist.empty:
                    continue

                # ------------------------------------------------
                # TIME WEIGHT
                # ------------------------------------------------

                hist_age_days = (
                    (
                        current_date
                        -
                        hist[date_col]
                    )
                    .dt.days
                    .to_numpy(
                        dtype=np.int32
                    )
                )

                time_weight = (
                    window_days
                    -
                    hist_age_days
                ).astype(
                    np.float64
                )

                time_weight = np.clip(
                    time_weight,
                    1,
                    window_days
                )

                # ------------------------------------------------
                # FINAL WEIGHT
                # ------------------------------------------------

                if window_name == "monthly":

                    label_weight = (
                        hist[
                            label_weight_col
                        ]
                        .to_numpy(
                            dtype=np.float64
                        )
                    )

                    weights = (
                        time_weight
                        * label_weight
                    )

                else:

                    weights = time_weight

                hist["_profile_weight"] = (
                    weights
                )

                # ------------------------------------------------
                # GROUP
                # ------------------------------------------------

                grouped = hist.groupby(
                    group_cols,
                    dropna=False,
                    sort=False
                )

                for key, group in grouped:

                    if not isinstance(
                        key,
                        tuple
                    ):
                        key = (key,)

                    values = (
                        group[price_col]
                        .to_numpy(
                            dtype=np.float64
                        )
                    )

                    weights_group = (
                        group[
                            "_profile_weight"
                        ]
                        .to_numpy(
                            dtype=np.float64
                        )
                    )

                    stats = weighted_stats(
                        values,
                        weights_group
                    )

                    row = {
                        date_col: current_date
                    }

                    for col, value in zip(
                        group_cols,
                        key
                    ):
                        row[col] = value

                    for metric, value in (
                        stats.items()
                    ):

                        row[
                            f"price_"
                            f"{level_name}_"
                            f"{window_name}_"
                            f"{metric}"
                        ] = value

                    output_rows.append(
                        row
                    )

        if not output_rows:
            return pd.DataFrame()

        result = pd.DataFrame(
            output_rows
        )

        # -----------------------------------------------------
        # UNIQUE KEY
        # -----------------------------------------------------

        merge_keys = (
            group_cols
            + [date_col]
        )

        result = (
            result
            .drop_duplicates(
                subset=merge_keys,
                keep="last"
            )
            .reset_index(drop=True)
        )

        assert not result.duplicated(
            subset=merge_keys
        ).any()

        # -----------------------------------------------------
        # MEMORY OPTIMIZATION
        # -----------------------------------------------------

        value_cols = [
            c
            for c in result.columns
            if c not in merge_keys
        ]

        for col in value_cols:

            if col.endswith(
                "_sample_count"
            ):

                result[col] = pd.to_numeric(
                    result[col],
                    errors="coerce"
                ).fillna(0).astype(
                    "int32"
                )

            else:

                result[col] = pd.to_numeric(
                    result[col],
                    errors="coerce"
                ).astype(
                    "float32"
                )

        return result

    # =========================================================
    # 7. BUILD L1-L4 PROFILES
    # =========================================================

    profile_tables = {}

    for level_name, group_cols in (
        group_defs.items()
    ):

        profile_tables[level_name] = (
            build_profile_table(
                profile_source,
                level_name,
                group_cols
            )
        )

    # =========================================================
    # 8. MERGE L1-L4
    # =========================================================

    for level_name, group_cols in (
        group_defs.items()
    ):

        profile_table = (
            profile_tables[level_name]
        )

        if profile_table.empty:
            continue

        merge_keys = (
            group_cols
            + [date_col]
        )

        value_cols = [
            c
            for c in profile_table.columns
            if c not in merge_keys
        ]

        assert not profile_table.duplicated(
            subset=merge_keys
        ).any()

        before_rows = len(df)

        df = df.merge(
            profile_table[
                merge_keys
                + value_cols
            ],
            on=merge_keys,
            how="left",
            sort=False,
            validate="many_to_one",
        )

        after_rows = len(df)

        if before_rows != after_rows:

            raise ValueError(
                f"Row count changed at {level_name}: "
                f"{before_rows} -> {after_rows}"
            )

        for col in value_cols:

            if col.endswith(
                "_sample_count"
            ):

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).fillna(0).astype(
                    "int32"
                )

            else:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).astype(
                    "float32"
                )

    # =========================================================
    # 9. PARENT SOURCE
    # =========================================================

    parent_cols = [
        "real_brand",
        "real_model",
        "is_copy",
    ]

    parent_source = profile_source[
        parent_cols
        + [
            date_col,
            price_col,
            label_weight_col,
        ]
    ].copy()

    parent_source = (
        parent_source
        .sort_values(
            date_col,
            kind="mergesort"
        )
        .reset_index(drop=True)
    )

    parent_date_array = (
        parent_source[date_col]
        .values
        .astype("datetime64[ns]")
    )

    parent_unique_dates = np.unique(
        parent_date_array
    )

    # =========================================================
    # 10. PARENT PROFILE BUILDER
    # =========================================================

    def build_parent_profile(
        source,
        source_dates,
        unique_dates,
        window_days,
        prefix,
    ):

        output_rows = []

        for current_date64 in unique_dates:

            current_date = pd.Timestamp(
                current_date64
            )

            right_idx = np.searchsorted(
                source_dates,
                current_date64,
                side="left"
            )

            if right_idx == 0:
                continue

            window_start64 = (
                current_date64
                -
                np.timedelta64(
                    window_days,
                    "D"
                )
            )

            left_idx = np.searchsorted(
                source_dates,
                window_start64,
                side="left"
            )

            if left_idx >= right_idx:
                continue

            hist = source.iloc[
                left_idx:right_idx
            ].copy()

            if hist.empty:
                continue

            age_days = (
                current_date
                -
                hist[date_col]
            ).dt.days.to_numpy(
                dtype=np.int32
            )

            time_weight = (
                window_days
                -
                age_days
            ).astype(
                np.float64
            )

            time_weight = np.clip(
                time_weight,
                1,
                window_days
            )

            if window_days == 30:

                weights = (
                    time_weight
                    *
                    hist[
                        label_weight_col
                    ]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

            else:

                weights = time_weight

            hist["_parent_weight"] = (
                weights
            )

            grouped = hist.groupby(
                parent_cols,
                dropna=False,
                sort=False
            )

            for key, group in grouped:

                if not isinstance(
                    key,
                    tuple
                ):
                    key = (key,)

                median = weighted_quantile(
                    group[price_col]
                    .to_numpy(
                        dtype=np.float64
                    ),
                    group[
                        "_parent_weight"
                    ].to_numpy(
                        dtype=np.float64
                    ),
                    0.50
                )

                row = {
                    date_col: current_date,

                    f"parent_{prefix}_median":
                        median,

                    f"parent_{prefix}_sample_count":
                        len(group),
                }

                for col, value in zip(
                    parent_cols,
                    key
                ):
                    row[col] = value

                output_rows.append(
                    row
                )

        if not output_rows:
            return pd.DataFrame()

        result = pd.DataFrame(
            output_rows
        )

        merge_keys = (
            parent_cols
            + [date_col]
        )

        result = (
            result
            .drop_duplicates(
                subset=merge_keys,
                keep="last"
            )
            .reset_index(drop=True)
        )

        assert not result.duplicated(
            subset=merge_keys
        ).any()

        result[
            f"parent_{prefix}_median"
        ] = pd.to_numeric(
            result[
                f"parent_{prefix}_median"
            ],
            errors="coerce"
        ).astype(
            "float32"
        )

        result[
            f"parent_{prefix}_sample_count"
        ] = pd.to_numeric(
            result[
                f"parent_{prefix}_sample_count"
            ],
            errors="coerce"
        ).fillna(0).astype(
            "int32"
        )

        return result

    # =========================================================
    # 11. PARENT MONTHLY
    # =========================================================

    parent_monthly = build_parent_profile(
        parent_source,
        parent_date_array,
        parent_unique_dates,
        window_days=30,
        prefix="monthly",
    )

    if not parent_monthly.empty:

        before_rows = len(df)

        df = df.merge(
            parent_monthly,
            on=parent_cols + [date_col],
            how="left",
            sort=False,
            validate="many_to_one",
        )

        if len(df) != before_rows:

            raise ValueError(
                "Row count changed after "
                "parent monthly merge."
            )

    # =========================================================
    # 12. PARENT WEEKLY
    # =========================================================

    parent_weekly = build_parent_profile(
        parent_source,
        parent_date_array,
        parent_unique_dates,
        window_days=7,
        prefix="weekly",
    )

    if not parent_weekly.empty:

        before_rows = len(df)

        df = df.merge(
            parent_weekly,
            on=parent_cols + [date_col],
            how="left",
            sort=False,
            validate="many_to_one",
        )

        if len(df) != before_rows:

            raise ValueError(
                "Row count changed after "
                "parent weekly merge."
            )

    # =========================================================
    # 13. PRICE PROFILE COLUMNS -> NUMERIC
    # =========================================================

    profile_columns = [
        c
        for c in df.columns
        if c.startswith("price_l")
    ]

    for col in profile_columns:

        if col.endswith(
            "_sample_count"
        ):

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0).astype(
                "int32"
            )

        else:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype(
                "float32"
            )

    # =========================================================
    # 14. MONTHLY MEDIAN / SUPPORT
    # =========================================================

    monthly_median_cols = {
        "l1":
            "price_l1_monthly_median",

        "l2":
            "price_l2_monthly_median",

        "l3":
            "price_l3_monthly_median",

        "l4":
            "price_l4_monthly_median",
    }

    monthly_count_cols = {
        "l1":
            "price_l1_monthly_sample_count",

        "l2":
            "price_l2_monthly_sample_count",

        "l3":
            "price_l3_monthly_sample_count",

        "l4":
            "price_l4_monthly_sample_count",
    }

    # =========================================================
    # 15. WEEKLY MEDIAN / SUPPORT
    # =========================================================

    weekly_median_cols = {
        "l1":
            "price_l1_weekly_median",

        "l2":
            "price_l2_weekly_median",

        "l3":
            "price_l3_weekly_median",

        "l4":
            "price_l4_weekly_median",
    }

    weekly_count_cols = {
        "l1":
            "price_l1_weekly_sample_count",

        "l2":
            "price_l2_weekly_sample_count",

        "l3":
            "price_l3_weekly_sample_count",

        "l4":
            "price_l4_weekly_sample_count",
    }

    # =========================================================
    # 16. HIERARCHICAL MONTHLY BASELINE
    # =========================================================

    monthly_baseline = pd.Series(
        np.nan,
        index=df.index,
        dtype="float32"
    )

    monthly_level = pd.Series(
        "",
        index=df.index,
        dtype="object"
    )

    for level in (
        "l1",
        "l2",
        "l3",
        "l4",
    ):

        median = pd.to_numeric(
            df[
                monthly_median_cols[level]
            ],
            errors="coerce"
        )

        support = pd.to_numeric(
            df[
                monthly_count_cols[level]
            ],
            errors="coerce"
        )

        valid = (
            support
            >= min_monthly_support
        ) & (
            median.notna()
        )

        take = (
            monthly_baseline.isna()
            &
            valid
        )

        monthly_baseline.loc[take] = (
            median.loc[take]
        )

        monthly_level.loc[take] = level

    # =========================================================
    # 17. PARENT MONTHLY FALLBACK
    # =========================================================

    parent_monthly_median = pd.to_numeric(
        df[
            "parent_monthly_median"
        ],
        errors="coerce"
    )

    parent_monthly_count = pd.to_numeric(
        df[
            "parent_monthly_sample_count"
        ],
        errors="coerce"
    ).fillna(0)

    parent_monthly_valid = (
        parent_monthly_count
        >= min_monthly_support
    ) & (
        parent_monthly_median.notna()
    )

    take_parent = (
        monthly_baseline.isna()
        &
        parent_monthly_valid
    )

    monthly_baseline.loc[take_parent] = (
        parent_monthly_median.loc[take_parent]
    )

    monthly_level.loc[take_parent] = (
        "parent"
    )

    # =========================================================
    # 18. HIERARCHICAL WEEKLY BASELINE
    # =========================================================

    weekly_baseline = pd.Series(
        np.nan,
        index=df.index,
        dtype="float32"
    )

    weekly_level = pd.Series(
        "",
        index=df.index,
        dtype="object"
    )

    for level in (
        "l1",
        "l2",
        "l3",
        "l4",
    ):

        median = pd.to_numeric(
            df[
                weekly_median_cols[level]
            ],
            errors="coerce"
        )

        support = pd.to_numeric(
            df[
                weekly_count_cols[level]
            ],
            errors="coerce"
        )

        valid = (
            support
            >= min_weekly_support
        ) & (
            median.notna()
        )

        take = (
            weekly_baseline.isna()
            &
            valid
        )

        weekly_baseline.loc[take] = (
            median.loc[take]
        )

        weekly_level.loc[take] = level

    # =========================================================
    # 19. PARENT WEEKLY FALLBACK
    # =========================================================

    parent_weekly_median = pd.to_numeric(
        df[
            "parent_weekly_median"
        ],
        errors="coerce"
    )

    parent_weekly_count = pd.to_numeric(
        df[
            "parent_weekly_sample_count"
        ],
        errors="coerce"
    ).fillna(0)

    parent_weekly_valid = (
        parent_weekly_count
        >= min_weekly_support
    ) & (
        parent_weekly_median.notna()
    )

    take_parent_weekly = (
        weekly_baseline.isna()
        &
        parent_weekly_valid
    )

    weekly_baseline.loc[
        take_parent_weekly
    ] = (
        parent_weekly_median.loc[
            take_parent_weekly
        ]
    )

    weekly_level.loc[
        take_parent_weekly
    ] = "parent"

    # =========================================================
    # 20. PROFILE SUPPORT
    # =========================================================

    profile_support = pd.Series(
        0,
        index=df.index,
        dtype="int32"
    )

    for level in (
        "l1",
        "l2",
        "l3",
        "l4",
    ):

        support = pd.to_numeric(
            df[
                monthly_count_cols[level]
            ],
            errors="coerce"
        ).fillna(0)

        take = (
            monthly_level == level
        )

        profile_support.loc[take] = (
            support.loc[take]
            .astype("int32")
        )

    # Parent
    take_parent = (
        monthly_level == "parent"
    )

    profile_support.loc[take_parent] = (
        parent_monthly_count.loc[
            take_parent
        ]
        .astype("int32")
    )

    # =========================================================
    # 21. RELATIVE MOMENTUM
    # =========================================================
    #
    # در نسخه قبلی:
    # weekly_target_ratio - monthly_target_ratio
    #
    # اینجا:
    # (weekly_price - monthly_price) / monthly_price
    #
    # این مقیاس برای قیمت مستقل از واحد پول است.
    # =========================================================

    momentum_relative = (
        (
            weekly_baseline
            -
            monthly_baseline
        )
        /
        monthly_baseline.clip(
            lower=1e-6
        )
    )

    momentum_relative = (
        momentum_relative
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
        .clip(
            -momentum_clip,
            momentum_clip
        )
        .astype("float32")
    )

    # =========================================================
    # 22. SUPPORT COMPONENT
    # =========================================================

    support_component = (
        np.log1p(
            profile_support
            .clip(lower=0)
            .astype(float)
        )
        /
        np.log1p(50)
    )

    support_component = (
        support_component
        .clip(0, 1)
        .astype("float32")
    )

    # =========================================================
    # 23. CONSISTENCY COMPONENT
    # =========================================================
    #
    # اختلاف نسبی weekly و monthly
    # =========================================================

    baseline_gap_relative = (
        (
            weekly_baseline
            -
            monthly_baseline
        ).abs()
        /
        monthly_baseline.clip(
            lower=1e-6
        )
    )

    consistency_component = np.exp(
        -baseline_gap_relative
        / 0.10
    )

    consistency_component = (
        consistency_component
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
        .clip(0, 1)
        .astype("float32")
    )

    # =========================================================
    # 24. BASELINE CONFIDENCE
    # =========================================================

    baseline_confidence = (
        0.65
        *
        support_component
        +
        0.35
        *
        consistency_component
    )

    baseline_confidence = (
        baseline_confidence
        .clip(0, 1)
        .astype("float32")
    )

    # =========================================================
    # 25. ADAPTIVE ALPHA
    # =========================================================
    #
    # momentum_relative الان مثلاً 0.08 یعنی +8%.
    # =========================================================

    momentum_strength = (
        momentum_relative.abs()
        /
        momentum_clip
    ).clip(
        0,
        1
    )

    support_factor = (
        np.log1p(
            profile_support
            .clip(lower=0)
            .astype(float)
        )
        /
        np.log1p(50)
    ).clip(
        0,
        1
    )

    alpha = (
        0.15
        +
        0.60
        *
        momentum_strength
    )

    alpha = (
        alpha
        *
        support_factor
    )

    alpha = alpha.clip(
        0,
        1
    ).astype("float32")

    # =========================================================
    # 26. ADAPTIVE PRICE BASELINE
    # =========================================================

    adaptive_baseline = (
        (1 - alpha)
        *
        monthly_baseline.astype(
            "float32"
        )
        +
        alpha
        *
        weekly_baseline.astype(
            "float32"
        )
    )

    # monthly missing -> weekly
    adaptive_baseline = (
        adaptive_baseline
        .combine_first(
            weekly_baseline
        )
    )

    # weekly missing -> monthly
    adaptive_baseline = (
        adaptive_baseline
        .combine_first(
            monthly_baseline
        )
    )

    adaptive_baseline = pd.to_numeric(
        adaptive_baseline,
        errors="coerce"
    ).astype("float32")

    # =========================================================
    # 27. SAVE BASELINE FEATURES
    # =========================================================

    df["baseline_monthly"] = (
        monthly_baseline
        .astype("float32")
    )

    df["baseline_weekly"] = (
        weekly_baseline
        .astype("float32")
    )

    df["baseline_monthly_level"] = (
        monthly_level
    )

    df["baseline_weekly_level"] = (
        weekly_level
    )

    df["profile_support"] = (
        profile_support
        .astype("int32")
    )

    # حفظ نسبت momentum
    df["momentum_7d_30d"] = (
        momentum_relative
        .astype("float32")
    )

    df["baseline_alpha"] = (
        alpha
        .astype("float32")
    )

    df["baseline_confidence"] = (
        baseline_confidence
        .astype("float32")
    )

    df["adaptive_baseline"] = (
        adaptive_baseline
        .astype("float32")
    )

    # =========================================================
    # 28. OVERWRITE / KEEP OLD BASELINE
    # =========================================================

    if "baseline" in df.columns:

        df["baseline_old"] = pd.to_numeric(
            df["baseline"],
            errors="coerce"
        ).astype("float32")

    # baseline جدید = قیمت
    df["baseline"] = (
        df["adaptive_baseline"]
        .astype("float32")
    )

    # =========================================================
    # 29. PRICE RESIDUAL
    # =========================================================

    df["adaptive_residual"] = (
        pd.to_numeric(
            df[price_col],
            errors="coerce"
        ).astype("float32")
        -
        df["adaptive_baseline"]
        .astype("float32")
    )

    # =========================================================
    # 30. LOG PRICE RESIDUAL
    # =========================================================

    df["adaptive_residual_log"] = (
        np.log(
            pd.to_numeric(
                df[price_col],
                errors="coerce"
            )
            .clip(lower=1e-6)
        )
        -
        np.log(
            df["adaptive_baseline"]
            .clip(lower=1e-6)
        )
    ).astype("float32")

    df["price_l1_monthly_median_residual_log"] = (
        np.log(
            pd.to_numeric(
                df["price"],
                errors="coerce"
            )
            .clip(lower=1e-6)
        )
        -
        np.log(
            df["price_l1_monthly_median"]
            .clip(lower=1e-6)
        )
    ).astype("float32")

    # =========================================================
    # 31. ADDITIONAL MOMENTUM FEATURES
    # =========================================================

    # اختلاف مطلق قیمت
    df["momentum_l1_abs"] = (
        df["price_l1_weekly_median"]
        -
        df["price_l1_monthly_median"]
    ).astype("float32")

    # اختلاف نسبی قیمت
    df["momentum_relative_l1"] = (
        (
            df["price_l1_weekly_median"]
            -
            df["price_l1_monthly_median"]
        )
        /
        df["price_l1_monthly_median"]
        .clip(lower=1e-6)
    ).astype("float32")

    # momentum بر اساس IQR
    df["momentum_z_score"] = (
        (
            df["price_l1_weekly_median"]
            -
            df["price_l1_monthly_median"]
        )
        /
        df["price_l1_monthly_iqr"]
        .clip(lower=1e-3)
    ).astype("float32")

    # =========================================================
    # 32. FINAL SORT BACK TO ORIGINAL ORDER
    # =========================================================

    df = (
        df
        .sort_values(
            "_profile_row_id"
        )
        .reset_index(drop=True)
    )

    # =========================================================
    # 33. FINAL ASSERTIONS
    # =========================================================

    if len(df) != original_len:

        raise ValueError(
            f"FINAL ROW COUNT CHANGED: "
            f"{original_len} -> {len(df)}"
        )

    assert len(df) == original_len

    assert (
        df["_profile_row_id"]
        .duplicated()
        .sum()
        == 0
    )

    # =========================================================
    # 34. FINAL REPORT
    # =========================================================

    print("\n")
    print("=" * 70)
    print("FINAL PRICE PROFILE / ADAPTIVE BASELINE REPORT")
    print("=" * 70)

    print(
        f"Rows                        : "
        f"{len(df):,}"
    )

    print(
        f"Profile source rows         : "
        f"{len(profile_source):,}"
    )

    print(
        f"Adaptive baseline non-null  : "
        f"{df['adaptive_baseline'].notna().sum():,}"
    )

    print(
        f"Adaptive baseline null      : "
        f"{df['adaptive_baseline'].isna().sum():,}"
    )

    print(
        f"Adaptive residual non-null  : "
        f"{df['adaptive_residual'].notna().sum():,}"
    )

    print(
        f"Adaptive residual log non-null: "
        f"{df['adaptive_residual_log'].notna().sum():,}"
    )

    print(
        f"Support >= {min_monthly_support}            : "
        f"{(df['profile_support'] >= min_monthly_support).sum():,}"
    )

    print("\nMonthly baseline level:")
    print(
        df[
            "baseline_monthly_level"
        ].value_counts(
            dropna=False
        )
    )

    print("\nWeekly baseline level:")
    print(
        df[
            "baseline_weekly_level"
        ].value_counts(
            dropna=False
        )
    )

    print("\nAdaptive baseline:")
    print(
        df[
            "adaptive_baseline"
        ].describe()
    )

    print("\nBaseline confidence:")
    print(
        df[
            "baseline_confidence"
        ].describe()
    )

    print("\nProfile support:")
    print(
        df[
            "profile_support"
        ].describe()
    )

    print("\nRelative momentum:")
    print(
        df[
            "momentum_7d_30d"
        ].describe()
    )

    print("\nNaN count:")
    print(
        df[
            [
                "adaptive_baseline",
                "adaptive_residual",
                "adaptive_residual_log",
                'price_l1_monthly_median_residual_log',
                "baseline_confidence",
                "profile_support",
                "momentum_7d_30d",
                "baseline_alpha",
            ]
        ].isna().sum()
    )

    # =========================================================
    # 35. REMOVE INTERNAL COLUMN
    # =========================================================

    df = df.drop(
        columns="_profile_row_id"
    )

    return df
