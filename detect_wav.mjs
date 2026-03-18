/**
 * Detect if a WAV file is "fixed" (Windows-compatible PCM 16-bit)
 * or "compacted" (compressed/other format that may not play on Windows).
 *
 * Usage:
 *   node detect_wav.mjs <file.wav> [<file2.wav> ...]
 *   node detect_wav.mjs <directory>
 *
 * Output: FIXED or COMPACTED for each file, with format details.
 */

import { createReadStream } from "node:fs"
import { existsSync, statSync } from "node:fs"
import { join } from "node:path"
import { readdirSync } from "node:fs"

const WAV_FORMAT_PCM = 0x0001
const WAV_FORMAT_IEEE_FLOAT = 0x0003
const WAV_FORMAT_ALAW = 0x0006
const WAV_FORMAT_MULAW = 0x0007
const WAV_FORMAT_MS_ADPCM = 0x0002
const WAV_FORMAT_IMA_ADPCM = 0x0011

const FORMAT_NAMES = {
  [WAV_FORMAT_PCM]: "PCM",
  [WAV_FORMAT_IEEE_FLOAT]: "IEEE Float",
  [WAV_FORMAT_ALAW]: "A-law",
  [WAV_FORMAT_MULAW]: "μ-law",
  [WAV_FORMAT_MS_ADPCM]: "MS ADPCM",
  [WAV_FORMAT_IMA_ADPCM]: "IMA ADPCM",
}

function readU16LE(buf, offset) {
  return buf.readUInt16LE(offset)
}

function readU32LE(buf, offset) {
  return buf.readUInt32LE(offset)
}

function findFmtChunk(buffer) {
  if (buffer.length < 12) return null
  if (buffer.toString("ascii", 0, 4) !== "RIFF") return null
  if (buffer.toString("ascii", 8, 12) !== "WAVE") return null

  let offset = 12
  while (offset + 8 <= buffer.length) {
    const chunkId = buffer.toString("ascii", offset, offset + 4)
    const chunkSize = readU32LE(buffer, offset + 4)
    if (chunkId === "fmt ") {
      return { offset: offset + 8, size: chunkSize }
    }
    offset += 8 + chunkSize
  }
  return null
}

function parseWavFormat(filePath) {
  return new Promise((resolve, reject) => {
    const stream = createReadStream(filePath, { start: 0, end: 4095 })
    const chunks = []
    stream.on("data", (chunk) => chunks.push(chunk))
    stream.on("end", () => {
      const buffer = Buffer.concat(chunks)
      const fmt = findFmtChunk(buffer)
      if (!fmt) {
        resolve({ valid: false, error: "Invalid WAV or fmt chunk not found" })
        return
      }
      const { offset, size } = fmt
      if (offset + 16 > buffer.length) {
        resolve({ valid: false, error: "Truncated fmt chunk" })
        return
      }
      const formatTag = readU16LE(buffer, offset)
      const channels = readU16LE(buffer, offset + 2)
      const sampleRate = readU32LE(buffer, offset + 4)
      const byteRate = readU32LE(buffer, offset + 8)
      const blockAlign = readU16LE(buffer, offset + 12)
      const bitsPerSample = readU16LE(buffer, offset + 14)

      resolve({
        valid: true,
        formatTag,
        channels,
        sampleRate,
        byteRate,
        blockAlign,
        bitsPerSample,
        formatName: FORMAT_NAMES[formatTag] ?? `0x${formatTag.toString(16)}`,
      })
    })
    stream.on("error", reject)
  })
}

function isFixed(info) {
  if (!info.valid) return false
  return info.formatTag === WAV_FORMAT_PCM && info.bitsPerSample === 16
}

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

async function main() {
  const args = process.argv.slice(2)
  if (args.length === 0) {
    console.error("Usage: node detect_wav.mjs <file.wav> [<file2.wav> ...]")
    process.exit(1)
  }

  const wavFiles = collectWavFiles(args)
  if (wavFiles.length === 0) {
    console.log("No WAV files found.")
    process.exit(0)
  }

  console.log(`Checking ${wavFiles.length} WAV file(s)...\n`)
  console.log("  Status     Format        Ch  Rate    Bits  File")
  console.log("  " + "-".repeat(70))

  let fixedCount = 0
  let compactedCount = 0

  for (const wav of wavFiles) {
    const base = wav.replace(/\.wav$/i, "").replace(/^.*[/\\]/, "") + ".wav"
    try {
      const info = await parseWavFormat(wav)
      if (!info.valid) {
        console.log(`  ERROR      ${info.error?.padEnd(12) || "?"}              ${base}`)
        continue
      }
      const status = isFixed(info) ? "FIXED" : "COMPACTED"
      if (status === "FIXED") fixedCount++
      else compactedCount++

      const formatStr = info.formatName.padEnd(12)
      const chStr = String(info.channels).padStart(2)
      const rateStr = String(info.sampleRate).padStart(6)
      const bitsStr = String(info.bitsPerSample).padStart(4)
      console.log(`  ${status.padEnd(11)} ${formatStr} ${chStr} ${rateStr} ${bitsStr}  ${base}`)
    } catch (err) {
      console.log(`  ERROR      ${err.message.padEnd(12)}              ${base}`)
    }
  }

  console.log("  " + "-".repeat(70))
  console.log(`\n  FIXED: ${fixedCount} (Windows-compatible)  |  COMPACTED: ${compactedCount} (run fix_wav to convert)`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
