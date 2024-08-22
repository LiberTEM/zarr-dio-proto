# Writing to zarr arrays using `O_DIRECT`

Writing large amounts of data onto a fast medium (i.e. NVMe SSD, or a RAID of
those) can sometimes be sped up by using direct I/O (`O_DIRECT`). This repo
tries to compare writing a 32GiB zarr array using the built-in
`FilesystemStore` of `zarrs` with a manual writer implementation using
direct I/O.

NOTE: this is only a prototype, and not meant to be used for other purposes
than benchmarking.

## Usage

`cargo run --release -- /path/to/fast/storage/some-dataset/ --what=both`

## Example output

These numbers were obtained on: AMD EPYC 7F72, 2x KCM61VUL3T20 NVMe SSD (RAID 0).

Running the benchmark gives output like this:

```
filling data with randomness...
... done in 13.977061877s.
write_buffered_io took 37.868774784s
write_direct_io took 7.341378322s
```

It creates both a `/direct` and a `/buffered` array in the zarr dataset, which
can be compared for equality using:

`cargo run --release -- /path/to/fast/storage/some-dataset/ --what=compare`
