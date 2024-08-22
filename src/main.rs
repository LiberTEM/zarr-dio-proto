use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs::OpenOptions,
    io::Write,
    mem::size_of,
    ops::{Deref, DerefMut},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    slice,
    sync::Arc,
    time::Instant,
};

use clap::Parser;
use ndarray::{ArrayView3, Axis, Slice};
use rand::RngCore;
use zarrs::{
    array::{Array, ArrayBuilder, FillValue}, array_subset::ArraySubset, storage::{data_key, store::FilesystemStore, ReadableWritableListableStorage, StoreKey}
};
use zerocopy::{FromBytes, AsBytes};

use nix::libc::O_DIRECT;

/// For O_DIRECT, we need a buffer that is aligned to the page size and is a
/// multiple of the page size.
pub struct PageAlinedBuffer {
    buf: *mut u8,
    layout: Layout,
}

impl PageAlinedBuffer {
    pub fn new(size: usize) -> Self {
        let align = page_size::get();
        let pad_size = (align - (size % align)) % align;
        let padded_size = size + pad_size;
        let layout = Layout::from_size_align(padded_size, align).unwrap();
        assert!(layout.size() > 0);
        let buf = unsafe { System.alloc_zeroed(layout) };

        Self { buf, layout }
    }
}

impl Deref for PageAlinedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.buf, self.layout.size()) }
    }
}

impl DerefMut for PageAlinedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.buf, self.layout.size()) }
    }
}

impl Drop for PageAlinedBuffer {
    fn drop(&mut self) {
        unsafe { System.dealloc(self.buf, self.layout) }
    }
}

/// The chunk size in the first dimension of our array
const CHUNK: usize = 16;
const SIDE: u64 = 512;
const SHAPE: [u64; 3] = [65536, SIDE, SIDE];

/// Write the array using the built-in `FilesystemStore` of `zarrs`
fn write_buffered_io(save_path: &Path, array_path: &str, input_data: &PageAlinedBuffer) {
    let input_data = ArrayView3::from_shape([65536, SIDE as usize, SIDE as usize], u16::slice_from(input_data).unwrap()).unwrap();
    let store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(save_path).unwrap());
    let chunk_grid = vec![CHUNK as u64, SIDE, SIDE];

    let array = ArrayBuilder::new(
        SHAPE.to_vec(),
        zarrs::array::DataType::UInt16,
        chunk_grid.try_into().unwrap(),
        FillValue::from(0u16),
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

fn key_to_fspath(save_path: &Path, key: &StoreKey) -> PathBuf {
    let mut path = save_path.to_owned();
    if !key.as_str().is_empty() {
        path.push(key.as_str().strip_prefix('/').unwrap_or(key.as_str()));
    }
    path
}

/// Write the array using a fast-path O_DIRECT writer
fn write_direct_io(save_path: &Path, array_path: &str, input_data: &PageAlinedBuffer) {
    let input_data = ArrayView3::from_shape([65536, SIDE as usize, SIDE as usize], u16::slice_from(input_data).unwrap()).unwrap();
    let store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(save_path).unwrap());
    let chunk_grid = vec![CHUNK as u64, SIDE, SIDE];

    let array = ArrayBuilder::new(
        SHAPE.to_vec(),
        zarrs::array::DataType::UInt16,
        chunk_grid.try_into().unwrap(),
        FillValue::from(0u16),
    )
    .dimension_names(["i", "Ky", "Kx"].into())
    .build(Arc::clone(&store), array_path)
    .unwrap();
    array.store_metadata().unwrap();

    let t0 = Instant::now();

    for i in 0..(65536 / CHUNK as u64) {
        // NOTE: the slicing here happens to align to the page size
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
        let align = page_size::get() as u64;
        let pad_size = (align - (cutoff % align)) % align;
        let aligned_cutoff = (cutoff + pad_size) as usize;
        assert!(aligned_cutoff >= cutoff as usize);
        assert!(aligned_cutoff % align as usize == 0);

        // Write
        file.write_all(data.as_bytes()).unwrap();

        // We may have written more because of page-size alignment; truncate.
        file.set_len(cutoff).unwrap();
    }

    eprintln!("write_direct_io took {:?}", t0.elapsed());
}

#[derive(Default, Clone, Debug, clap::ValueEnum)]
enum RunWhat {
    Compare,
    Both,
    Buffered,
    #[default]
    Direct,
}

#[derive(clap::Parser)]
struct Args {
    save_prefix: PathBuf,

    #[arg(short, long)]
    what: RunWhat,
}

fn make_data() -> PageAlinedBuffer {
    let mut buf = PageAlinedBuffer::new(2 * 65536 * (SIDE * SIDE) as usize * size_of::<u16>());

    eprintln!("filling data with randomness...");
    let t0 = Instant::now();
    rand::thread_rng().fill_bytes(&mut buf);
    eprintln!("... done in {:?}.", t0.elapsed());

    buf
}

fn main() {
    let args = Args::parse();
    match args.what {
        RunWhat::Both => {
            let input_arr = make_data();
            write_buffered_io(&args.save_prefix, "/buffered", &input_arr);
            write_direct_io(&args.save_prefix, "/direct", &input_arr);
        }
        RunWhat::Direct => {
            let input_arr = make_data();
            write_direct_io(&args.save_prefix, "/direct", &input_arr);
        }
        RunWhat::Buffered => {
            let input_arr = make_data();
            write_buffered_io(&args.save_prefix, "/buffered", &input_arr);
        }
        RunWhat::Compare => {
            let store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&args.save_prefix).unwrap());
            let a_buf = Array::open(Arc::clone(&store), "/buffered").unwrap();
            let a_dir = Array::open(Arc::clone(&store), "/direct").unwrap();

            for i in 0..(65536 / CHUNK) {
                let subset: ArraySubset = ArraySubset::new_with_ranges(&[i as u64..i as u64 + CHUNK as u64, 0..SIDE, 0..SIDE]);
                let buf_bytes = a_buf.retrieve_array_subset(&subset).unwrap();
                let dir_bytes = a_dir.retrieve_array_subset(&subset).unwrap();
                assert_eq!(buf_bytes, dir_bytes);
            }
        }
    }
}
