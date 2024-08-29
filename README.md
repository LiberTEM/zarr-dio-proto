# Writing to zarr arrays using `O_DIRECT`

Writing large amounts of data onto a fast medium (i.e. NVMe SSD, or a RAID of
those) can sometimes be sped up by using direct I/O (`O_DIRECT`). This repo
tries to compare writing a 32GiB zarr array using the built-in
`FilesystemStore` of `zarrs` with a manual writer implementation using
direct I/O.

NOTE: this is only a prototype, and not meant to be used for other purposes
than benchmarking.

## Usage

`cargo run --release -- /path/to/fast/storage/some-dataset/ --what=all`

## Example output

These numbers were obtained on: AMD EPYC 7F72, 2x [KCM61VUL3T20](https://europe.kioxia.com/en-europe/business/ssd/enterprise-ssd/cm6-v.html) NVMe SSD (RAID 0).

Running the benchmark gives output like this:

```
filling data with randomness...
... done in 13.977061877s.
write_buffered_io took 37.868774784s
write_direct_io took 7.341378322s
```

It creates the following arrays in the zarr dataset:

- `/direct` written "manually" using `O_DIRECT` 
- `/buffered` written using normal buffered I/O
- `/direct_zarrs` written with the direct I/O support from [zarrs/58](https://github.com/LDeakin/zarrs/pull/58)
- `/direct_zarrs_encoded` written with the fast-path API from [zarrs/64](https://github.com/LDeakin/zarrs/pull/64)

They be compared for equality using:

`cargo run --release -- /path/to/fast/storage/some-dataset/ --what=compare`
