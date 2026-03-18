/**
 * Fix voice recorder WAV files for Windows compatibility.
 *
 * Converts non-standard WAV formats to standard PCM 16-bit WAV
 * that plays everywhere on Windows. Requires ffmpeg on PATH.
 *
 * Usage:
 *   node fix_wav.mjs <file.wav> [<file2.wav> ...]
 *   node fix_wav.mjs <directory>
 *
 * Options:
 *   --replace   Overwrite originals instead of creating *_fixed.wav
 */

import { execFile } from "node:child_process"
import { existsSync, renameSync, unlinkSync } from "node:fs"
import { dirname, join } from "node:path"
import { readdirSync, statSync } from "node:fs"

function collectWavFiles(paths) {
  const wavFiles = []
  for (const p of paths) {
    if (!existsSync(p)) {
      console.error(`Error: '${p}' does not exist.`)
      process.exit(1)
    }
    const stat = statSync(p)
    if (stat.isFile()) {
      if (p.toLowerCase().endsWith(".wav")) wavFiles.push(p)
      else console.log(`Skipping non-WAV: ${p}`)
    } else {
      for (const entry of walkDir(p)) {
        if (entry.toLowerCase().endsWith(".wav")) wavFiles.push(entry)
      }
    }
  }
  return wavFiles.sort()
}

function* walkDir(dir) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name)
    if (statSync(full).isDirectory()) {
      yield* walkDir(full)
    } else {
      yield full
    }
  }
}

function fixWav(inputPath, outputPath) {
  return new Promise((resolve) => {
    execFile(
      "ffmpeg",
      ["-hide_banner", "-loglevel", "error", "-i", inputPath, "-acodec", "pcm_s16le", "-y", outputPath],
      (err, _stdout, stderr) => {
        if (err) {
          if (err.code === "ENOENT") {
            console.error("  ffmpeg not found. Install ffmpeg and add it to PATH.")
          } else {
            console.error(`  ffmpeg ERROR: ${stderr?.trim() || err.message}`)
          }
          resolve(false)
        } else {
          resolve(true)
        }
      }
    )
  })
}

async function main() {
  const args = process.argv.slice(2)
  const replaceIdx = args.indexOf("--replace")
  const replace = replaceIdx >= 0
  if (replace) args.splice(replaceIdx, 1)

  if (args.length === 0) {
    console.error("Usage: node fix_wav.mjs <file.wav> [<file2.wav> ...] [--replace]")
    process.exit(1)
  }

  const wavFiles = collectWavFiles(args)
  if (wavFiles.length === 0) {
    console.log("No WAV files found.")
    process.exit(0)
  }

  console.log(`Fixing ${wavFiles.length} WAV file(s)...\n`)
  let ok = 0

  for (const wav of wavFiles) {
    const dir = dirname(wav)
    const base = wav.replace(/\.wav$/i, "").replace(/^.*[/\\]/, "")
    const outPath = replace ? join(dir, base + "_tmp.wav") : join(dir, base + "_fixed.wav")

    const success = await fixWav(wav, outPath)

    if (success) {
      if (replace) {
        try {
          renameSync(outPath, wav)
          console.log(`  OK: ${base}.wav`)
        } catch (e) {
          console.error(`  ERROR renaming: ${e.message}`)
          unlinkSync(outPath)
        }
      } else {
        console.log(`  OK: ${base}.wav -> ${base}_fixed.wav`)
      }
      ok++
    }
  }

  console.log(`\nDone. Fixed ${ok}/${wavFiles.length} files.`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
