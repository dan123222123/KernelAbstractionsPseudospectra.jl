# Producer/consumer contract for the cross-vendor σ dumps: repro writes one
# `cross_vendor_sigma_<backend>.csv` per vendor, compare globs them back — a drifted name
# fails SILENTLY as "0 vendors found". Tiny (not bench_common.jl): compare is a pure CSV
# diff with no package load.
cross_vendor_sigma_path(dir, tag) = joinpath(dir, "cross_vendor_sigma_$(tag).csv")
const CROSS_VENDOR_SIGMA_RX = r"^cross_vendor_sigma_(.+)\.csv$"
