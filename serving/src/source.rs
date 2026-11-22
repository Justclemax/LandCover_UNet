//! Pixel sources: the scene file on disk (positioned reads, nothing loaded up front) and
//! images posted by clients (in memory). Both are pixel-interleaved (H, W, C) float32.

use std::fs::File;
use std::io;
use std::path::Path;

pub trait Source: Sync {
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn channels(&self) -> usize;
    /// Read row `r`, columns `[c0, c1)`, into `out` (`(c1 - c0) * channels` values, HWC order).
    /// The caller guarantees the range is inside the source.
    fn read_row(&self, r: usize, c0: usize, c1: usize, out: &mut [f32]) -> io::Result<()>;
}

/// The scene as written by `landcover.export`. Each request reads only the rows it needs with
/// `pread`, so memory use does not depend on the scene size and readers never share a cursor.
pub struct FileScene {
    file: File,
    width: usize,
    height: usize,
    channels: usize,
}

impl FileScene {
    pub fn open(path: &Path, width: usize, height: usize, channels: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let expected = (width * height * channels * 4) as u64;
        let actual = file.metadata()?.len();
        if actual != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} : {actual} octets au lieu de {expected}", path.display()),
            ));
        }
        Ok(Self {
            file,
            width,
            height,
            channels,
        })
    }
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_at(file: &File, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !buf.is_empty() {
        let n = file.seek_read(buf, offset)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        buf = &mut buf[n..];
        offset += n as u64;
    }
    Ok(())
}

impl Source for FileScene {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn channels(&self) -> usize {
        self.channels
    }
    fn read_row(&self, r: usize, c0: usize, c1: usize, out: &mut [f32]) -> io::Result<()> {
        let n = (c1 - c0) * self.channels;
        let mut bytes = vec![0u8; n * 4];
        let offset = ((r * self.width + c0) * self.channels * 4) as u64;
        read_at(&self.file, &mut bytes, offset)?;
        let (words, _) = bytes.as_chunks::<4>();
        for (dst, word) in out[..n].iter_mut().zip(words) {
            *dst = f32::from_le_bytes(*word);
        }
        Ok(())
    }
}

/// An image sent by a client.
pub struct MemImage {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

impl Source for MemImage {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn channels(&self) -> usize {
        self.channels
    }
    fn read_row(&self, r: usize, c0: usize, c1: usize, out: &mut [f32]) -> io::Result<()> {
        let start = (r * self.width + c0) * self.channels;
        let n = (c1 - c0) * self.channels;
        out[..n].copy_from_slice(&self.data[start..start + n]);
        Ok(())
    }
}
