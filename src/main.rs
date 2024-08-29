use std::{
    fs::OpenOptions,
    io::Write,
    mem::size_of,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use bytes::BytesMut;
use clap::Parser;
use ndarray::{Array3, ArrayView3, Axis, Slice};
use rand::RngCore;
use zarrs::{
    array::{Array, ArrayBuilder, FillValue},
    array_subset::ArraySubset,
    storage::{data_key, store::{FilesystemStore, FilesystemStoreOptions}, ReadableWritableListableStorage, StoreKey},
};
use zerocopy::AsBytes;

use nix::libc::O_DIRECT;

//use tikv_jemallocator::Jemalloc;
//#[global_allocator]
//static GLOBAL: Jemalloc = Jemalloc;

/// The chunk size in the first dimension of our array
const CHUNK: usize = 16;
const SIDE: u64 = 512;
const SHAPE: [u64; 3] = [65536, SIDE, SIDE];

/// Write the array using the built-in `FilesystemStore` of `zarrs`
fn write_buffered_io(save_path: &Path, array_path: &str, input_data: &ArrayView3<u16>) {
    let store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(save_path).unwrap());
    let chunk_grid = vec![CHUNK as u64, SIDE, SIDE];

    let array = ArrayBuilder::new(
        SHAPE.to_vec(),
        zarrs::array::DataType::UInt16,
        chunk_grid.try_into().unwrap(),
        FillValue::from(7u16),
    )
    .dimension_names(["i", "Ky", "Kx"].into())
    .build(Arc::clone(&store), array_path)
    .unwrap();
    array.store_metadata().unwrap();

    let t0 = Instant::now();

    for i in 0..(65536 / CHUNK as u64) {
        let inp_slice = input_data.slice_axis(Axis(0), Slice::from(i as usize..i as usize + CHUNK));
        array
            .store_chunk_elements(&[i, 0, 0], inp_slice.as_slice().unwrap())
            .unwrap();
    }

    eprintln!("write_buffered_io took {:?}", t0.elapsed());
}

/// Write the array using the built-in `FilesystemStore` of `zarrs` with `direct_io` enabled.
fn write_direct_zarrs(save_path: &Path, array_path: &str, input_data: &ArrayView3<u16>) {
    let mut opts = FilesystemStoreOptions::default();
    opts.direct_io(true);
    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new_with_options(save_path, opts).unwrap());
    let chunk_grid = vec![CHUNK as u64, SIDE, SIDE];

    let array = ArrayBuilder::new(
        SHAPE.to_vec(),
        zarrs::array::DataType::UInt16,
        chunk_grid.try_into().unwrap(),
        FillValue::from(7u16),
    )
    .dimension_names(["i", "Ky", "Kx"].into())
    .build(Arc::clone(&store), array_path)
    .unwrap();
    array.store_metadata().unwrap();

    let t0 = Instant::now();

    for i in 0..(65536 / CHUNK as u64) {
        let inp_slice = input_data.slice_axis(Axis(0), Slice::from(i as usize..i as usize + CHUNK));
        array
            .store_chunk_elements(&[i, 0, 0], inp_slice.as_slice().unwrap())
            .unwrap();
    }

    eprintln!("write_direct_zarrs took {:?}", t0.elapsed());
}

/// For O_DIRECT, we need a buffer that is aligned to the page size and is a
/// multiple of the page size.
fn bytes_aligned(size: usize) -> BytesMut {
    let align = page_size::get();
    let mut bytes = BytesMut::with_capacity(size + align * 2);
    let offset = bytes.as_ptr().align_offset(align);
    bytes.split_off(offset)
}

/// Write the array using the built-in `FilesystemStore` of `zarrs` with `direct_io` enabled.
fn write_direct_zarrs_manual_encode(save_path: &Path, array_path: &str, input_data: &ArrayView3<u16>) {
    let mut opts = FilesystemStoreOptions::default();
    opts.direct_io(true);
    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new_with_options(save_path, opts).unwrap());
    let chunk_grid = vec![CHUNK as u64, SIDE, SIDE];

    let array = ArrayBuilder::new(
        SHAPE.to_vec(),
        zarrs::array::DataType::UInt16,
        chunk_grid.try_into().unwrap(),
        FillValue::from(7u16),
    )
    .dimension_names(["i", "Ky", "Kx"].into())
    .build(Arc::clone(&store), array_path)
    .unwrap();
    array.store_metadata().unwrap();

    let t0 = Instant::now();

    let mut buf: BytesMut = bytes_aligned(CHUNK *(SIDE * SIDE * 2) as usize);

    for i in 0..(65536 / CHUNK as u64) {
        assert!(buf.as_ptr().align_offset(page_size::get()) == 0, "a");

        let inp_slice = input_data.slice_axis(Axis(0), Slice::from(i as usize..i as usize + CHUNK));
        buf.clear();
        assert!(buf.as_ptr().align_offset(page_size::get()) == 0, "a.0");
        buf.extend_from_slice(inp_slice.as_slice().unwrap().as_bytes());

        assert!(buf.as_ptr().align_offset(page_size::get()) == 0, "b");

        let buf_frozen = buf.freeze();
        unsafe {
            array
            .store_encoded_chunk(&[i, 0, 0], buf_frozen.clone())
                .unwrap();
        }
        buf = buf_frozen.try_into_mut().unwrap();  // FIXME: handle the case where the buffer is still in use
    }

    eprintln!("write_direct_zarrs_manual_encode took {:?}", t0.elapsed());
}

fn key_to_fspath(save_path: &Path, key: &StoreKey) -> PathBuf {
    let mut path = save_path.to_owned();
    if !key.as_str().is_empty() {
        path.push(key.as_str().strip_prefix('/').unwrap_or(key.as_str()));
    }
    path
}

/// Write the array using a fast-path O_DIRECT writer
fn write_direct_io(save_path: &Path, array_path: &str, input_data: &ArrayView3<u16>) {
    let store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(save_path).unwrap());
    let chunk_grid = vec![CHUNK as u64, SIDE, SIDE];

    let array = ArrayBuilder::new(
        SHAPE.to_vec(),
        zarrs::array::DataType::UInt16,
        chunk_grid.try_into().unwrap(),
        FillValue::from(7u16),
    )
    .dimension_names(["i", "Ky", "Kx"].into())
    .build(Arc::clone(&store), array_path)
    .unwrap();
    array.store_metadata().unwrap();

    let t0 = Instant::now();

    let mut buf = bytes_aligned((SIDE * SIDE * 2) as usize * CHUNK);

    for i in 0..(65536 / CHUNK as u64) {
        let inp_slice = input_data.slice_axis(Axis(0), Slice::from(i as usize..i as usize + CHUNK));

        let chunk_indices = [i, 0, 0];
        let data = inp_slice.as_slice().unwrap();
        let key = data_key(array.path(), &chunk_indices, array.chunk_key_encoding());

        // Create directories
        let key_path = key_to_fspath(save_path, &key);
        if let Some(parent) = key_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).unwrap();
            }
        }

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(O_DIRECT)
            .open(key_path)
            .unwrap();

        // Only write as much as we have to
        let cutoff = SIDE * SIDE * CHUNK as u64 * size_of::<u16>() as u64;

        // Copy into aligned buffer:
        buf.clear();
        let data_bytes = data.as_bytes();
        let pad_size = data_bytes.len().next_multiple_of(page_size::get()) - data_bytes.len();
        buf.extend_from_slice(data_bytes);
        buf.extend(std::iter::repeat(0).take(pad_size));

        // Write
        file.write_all(&buf).unwrap();

        // We may have written more because of page-size alignment; truncate.
        file.set_len(cutoff).unwrap();
    }

    eprintln!("write_direct_io took {:?}", t0.elapsed());
}

#[derive(Default, Clone, Debug, clap::ValueEnum)]
enum RunWhat {
    Compare,
    All,
    Buffered,
    DirectZarrs,
    DirectZarrsEncoded,
    #[default]
    Direct,
}

#[derive(clap::Parser)]
struct Args {
    save_prefix: PathBuf,

    #[arg(short, long)]
    what: RunWhat,

    #[arg(short, long)]
    random: bool,
}

fn make_data(random: bool) -> Array3<u16> {
    let mut data = vec![0u16; 65536 * (SIDE * SIDE) as usize];
    let data_bytes = data.as_bytes_mut();

    if random {
        eprintln!("Generating random test data...");
        let t0 = Instant::now();
        rand::thread_rng().fill_bytes(data_bytes);
        eprintln!("... done in {:?}.", t0.elapsed());
    }

    Array3::from_shape_vec([65536, SIDE as usize, SIDE as usize], data).unwrap()
}

fn main() {
    let args = Args::parse();
    match args.what {
        RunWhat::All => {
            let input_arr = make_data(args.random);
            write_buffered_io(&args.save_prefix, "/buffered", &input_arr.view());
            write_direct_io(&args.save_prefix, "/direct", &input_arr.view());
            write_direct_zarrs_manual_encode(&args.save_prefix, "/direct_zarrs_encoded", &input_arr.view());
            write_direct_zarrs(&args.save_prefix, "/direct_zarrs", &input_arr.view());
        }
        RunWhat::Direct => {
            let input_arr = make_data(args.random);
            write_direct_io(&args.save_prefix, "/direct", &input_arr.view());
        }
        RunWhat::DirectZarrs => {
            let input_arr = make_data(args.random);
            write_direct_zarrs(&args.save_prefix, "/direct_zarrs", &input_arr.view());
        }
        RunWhat::DirectZarrsEncoded => {
            let input_arr = make_data(args.random);
            write_direct_zarrs_manual_encode(&args.save_prefix, "/direct_zarrs_encoded", &input_arr.view());
        }
        RunWhat::Buffered => {
            let input_arr = make_data(args.random);
            write_buffered_io(&args.save_prefix, "/buffered", &input_arr.view());
        }
        RunWhat::Compare => {
            let store: ReadableWritableListableStorage =
                Arc::new(FilesystemStore::new(&args.save_prefix).unwrap());
            let a_buf = Array::open(Arc::clone(&store), "/buffered").unwrap();
            let a_dir = Array::open(Arc::clone(&store), "/direct").unwrap();
            let a_dir_z = Array::open(Arc::clone(&store), "/direct_zarrs").unwrap();
            let a_dir_ze = Array::open(Arc::clone(&store), "/direct_zarrs_encoded").unwrap();

            let read_chunk = 1;
            for i in 0..(65536 / read_chunk) {
                let subset: ArraySubset = ArraySubset::new_with_ranges(&[
                    i as u64..i as u64 + read_chunk as u64,
                    0..SIDE,
                    0..SIDE,
                ]);
                let buf_bytes = a_buf.retrieve_array_subset(&subset).unwrap();
                let dir_bytes = a_dir.retrieve_array_subset(&subset).unwrap();
                let dir_z_bytes = a_dir_z.retrieve_array_subset(&subset).unwrap();
                let dir_ze_bytes = a_dir_ze.retrieve_array_subset(&subset).unwrap();
                assert_eq!(buf_bytes, dir_bytes);
                assert_eq!(buf_bytes, dir_z_bytes);
                assert_eq!(buf_bytes, dir_ze_bytes);
            }
        }
    }
}
