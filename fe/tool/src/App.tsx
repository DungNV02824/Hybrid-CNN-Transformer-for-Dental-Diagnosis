import { useState, useRef, useEffect } from 'react'
import DiagnosisDashboard from './DiagnosisDashboard'

// ─── Types ────────────────────────────────────────────────────────────────────
interface UploadedFile {
  file: File
  preview: string
}
type FilesState = Record<string, UploadedFile | null>
interface SlotDef {
  id: string
  label: string
  icon: React.ReactNode
}

// ─── Icons ───────────────────────────────────────────────────────────────────
const ICON_COLOR = "#6B7280"

function ToothLogoIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="white">
      <path d="M12 2C9 2 6 4.5 6 8c0 2.5 1 3.8 1.8 5.2C8.5 14.7 8.8 16 9 17.5c.2 1.2.8 2.5 2 2.5h2c1.2 0 1.8-1.3 2-2.5.2-1.5.5-2.8 1.2-4.3C17 11.8 18 10.5 18 8c0-3.5-3-6-6-6z" />
    </svg>
  )
}

function SearchIcon() {
  return (
    <svg width="16" height="16" fill="none" stroke="white" strokeWidth="2" viewBox="0 0 24 24">
      <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
    </svg>
  )
}

function BellIcon() {
  return (
    <svg width="18" height="18" fill="none" stroke="white" strokeWidth="2" viewBox="0 0 24 24">
      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg width="12" height="12" fill="none" stroke="white" strokeWidth="3" viewBox="0 0 24 24">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  )
}

function DiagnoseIcon() {
  return (
    <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
      <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

function BulkUploadIcon() {
  return (
    <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.2" viewBox="0 0 24 24">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )
}

// ─── Dental SVG Icons ─────────────────────────────────────────────────────────
function SkullLateralIcon() {
  return (
    <svg viewBox="0 0 100 100" width="80" height="80">
      <path d="M45 20 C20 20 15 45 20 60 C23 70 30 75 30 80 L35 80 C35 75 40 70 45 70 C45 65 50 65 55 70 L55 80 C60 85 65 85 70 80 L70 75 C70 70 65 65 65 60 C75 60 80 50 80 40 C80 25 70 20 45 20 Z" fill={ICON_COLOR} />
      <ellipse cx="68" cy="45" rx="3.5" ry="5" fill="white" />
      <path d="M62 60 L75 60 L75 68 L62 68 Z" fill="white" />
      <line x1="64" y1="60" x2="64" y2="68" stroke={ICON_COLOR} strokeWidth="1" />
      <line x1="68" y1="60" x2="68" y2="68" stroke={ICON_COLOR} strokeWidth="1" />
      <line x1="72" y1="60" x2="72" y2="68" stroke={ICON_COLOR} strokeWidth="1" />
      <line x1="62" y1="64" x2="75" y2="64" stroke={ICON_COLOR} strokeWidth="1" />
    </svg>
  )
}

function PanoramaIcon() {
  const teeth = [22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74]
  return (
    <svg viewBox="0 0 100 100" width="80" height="80">
      <rect x="15" y="35" width="70" height="30" rx="4" fill={ICON_COLOR} />
      <path d="M 20 45 Q 50 55 80 45" stroke="rgba(255,255,255,0.4)" strokeWidth="5" fill="none" />
      <path d="M 20 55 Q 50 65 80 55" stroke="rgba(255,255,255,0.4)" strokeWidth="5" fill="none" />
      <g fill="white">
        {teeth.map(x => <rect key={`t-${x}`} x={x} y="43" width="2" height="5" rx="1" />)}
      </g>
      <g fill="white">
        {teeth.map(x => <rect key={`b-${x}`} x={x} y="52" width="2" height="5" rx="1" />)}
      </g>
    </svg>
  )
}

function UpperJawIcon() {
  return (
    <svg viewBox="0 0 100 100" width="80" height="80" fill="none" stroke={ICON_COLOR} strokeWidth="1.5">
      <path d="M 25 80 C 15 40 35 20 50 20 C 65 20 85 40 75 80" />
      <path d="M 35 80 C 30 50 42 35 50 35 C 58 35 70 50 65 80" />
      <g strokeWidth="1" fill="white">
        <ellipse cx="45" cy="25" rx="4" ry="5" transform="rotate(-15 45 25)" />
        <ellipse cx="55" cy="25" rx="4" ry="5" transform="rotate(15 55 25)" />
        <ellipse cx="36" cy="30" rx="4" ry="5" transform="rotate(-30 36 30)" />
        <ellipse cx="64" cy="30" rx="4" ry="5" transform="rotate(30 64 30)" />
        <ellipse cx="29" cy="40" rx="5" ry="6" transform="rotate(-50 29 40)" />
        <ellipse cx="71" cy="40" rx="5" ry="6" transform="rotate(50 71 40)" />
        <ellipse cx="27" cy="55" rx="5" ry="7" transform="rotate(-70 27 55)" />
        <ellipse cx="73" cy="55" rx="5" ry="7" transform="rotate(70 73 55)" />
        <ellipse cx="28" cy="72" rx="6" ry="8" />
        <ellipse cx="72" cy="72" rx="6" ry="8" />
      </g>
    </svg>
  )
}

function LowerJawIcon() {
  return (
    <svg viewBox="0 0 100 100" width="80" height="80" fill="none" stroke={ICON_COLOR} strokeWidth="1.5">
      <path d="M 25 20 C 15 60 35 80 50 80 C 65 80 85 60 75 20" />
      <path d="M 35 20 C 30 50 42 65 50 65 C 58 65 70 50 65 20" />
      <path d="M 35 20 C 35 35 65 35 65 20" strokeDasharray="2 2" />
      <g strokeWidth="1" fill="white">
        <ellipse cx="45" cy="75" rx="3" ry="5" transform="rotate(15 45 75)" />
        <ellipse cx="55" cy="75" rx="3" ry="5" transform="rotate(-15 55 75)" />
        <ellipse cx="38" cy="70" rx="4" ry="5" transform="rotate(30 38 70)" />
        <ellipse cx="62" cy="70" rx="4" ry="5" transform="rotate(-30 62 70)" />
        <ellipse cx="31" cy="60" rx="5" ry="6" transform="rotate(50 31 60)" />
        <ellipse cx="69" cy="60" rx="5" ry="6" transform="rotate(-50 69 60)" />
        <ellipse cx="27" cy="45" rx="5" ry="7" transform="rotate(70 27 45)" />
        <ellipse cx="73" cy="45" rx="5" ry="7" transform="rotate(-70 73 45)" />
        <ellipse cx="28" cy="28" rx="6" ry="8" />
        <ellipse cx="72" cy="28" rx="6" ry="8" />
      </g>
    </svg>
  )
}

// ─── Slot Definitions ─────────────────────────────────────────────────────────
const xraySlots: SlotDef[] = [
  { id: 'lateral-skull', label: 'Phim sọ mặt nghiêng', icon: <SkullLateralIcon /> },
  { id: 'panorama', label: 'Phim Panorama', icon: <PanoramaIcon /> },
]

const clinicalSlots: SlotDef[] = [
  { id: 'upper-jaw', label: 'Ảnh mặt nhai hàm trên', icon: <UpperJawIcon /> },
  { id: 'lower-jaw', label: 'Ảnh mặt nhai hàm dưới', icon: <LowerJawIcon /> },
]

// ─── ImageSlot Component ──────────────────────────────────────────────────────
interface ImageSlotProps {
  slot: SlotDef
  uploaded: UploadedFile | null
  onUpload: (id: string, file: File) => void
  onRemove: (id: string) => void
}

function ImageSlot({ slot, uploaded, onUpload, onRemove }: ImageSlotProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) onUpload(slot.id, f)
  }

  return (
    <div
      className={`w-[180px] bg-white rounded-2xl flex flex-col p-5 relative transition-all duration-200
        ${uploaded
          ? 'border-[1.5px] border-teal-300 shadow-[0_4px_20px_rgba(46,196,182,0.12)]'
          : 'border-[1.5px] border-gray-100 hover:shadow-lg hover:-translate-y-0.5 hover:border-teal-200'
        }`}
    >
      {/* Label */}
      <div className="min-h-[36px] mb-4 flex items-center justify-center">
        <p className="text-[12.5px] text-center text-gray-600 font-semibold leading-snug line-clamp-2">
          {slot.label}
        </p>
      </div>

      {/* Preview or Icon */}
      <div className="w-full h-[100px] flex items-center justify-center mb-4 rounded-xl overflow-hidden bg-gradient-to-br from-gray-50 to-slate-100 border border-gray-100 relative">
        {uploaded ? (
          <>
            <img src={uploaded.preview} alt={slot.label} className="w-full h-full object-cover" />
            {/* Check badge */}
            {/* <div className="absolute top-1.5 left-1.5 w-5 h-5 bg-emerald-400 rounded-full flex items-center justify-center shadow-sm">
              <CheckIcon />
            </div> */}
            {/* Remove button */}
            <button
              onClick={(e) => { e.stopPropagation(); onRemove(slot.id) }}
              className="absolute top-1.5 right-1.5 w-5 h-5 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center text-xs font-bold shadow-sm transition-colors"
            >×</button>
          </>
        ) : (
          slot.icon
        )}
      </div>

      {/* Upload Button */}
      <button
        onClick={() => inputRef.current?.click()}
        className={`flex items-center justify-center gap-1.5 w-full py-2.5 rounded-full text-[12px] font-semibold transition-all duration-200
          ${uploaded
            ? 'bg-teal-50 border border-teal-300 text-teal-700 hover:bg-teal-100'
            : 'bg-gray-50 border-[1.5px] border-dashed border-gray-300 text-gray-500 hover:bg-teal-50 hover:border-teal-400 hover:text-teal-700 hover:border-solid'
          }`}
      >
        <UploadIcon />
        {uploaded ? 'Thay ảnh' : 'Tải lên'}
      </button>

      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleChange} />
    </div>
  )
}

// ─── Section Header Component ─────────────────────────────────────────────────
function SectionHeader({ title, subtitle, count, total }: { title: string; subtitle: string; count: number; total: number }) {
  return (
    <div className="flex items-center gap-4 mb-5 bg-white rounded-xl py-3 px-4 border border-gray-100 shadow-sm">
      <div className="w-1 h-10 rounded-full bg-gradient-to-b from-blue-500 to-teal-400 flex-shrink-0" />
      <div className="flex-1">
        <h2 className="font-bold text-[#0D1B2A] text-[15px] leading-tight">{title}</h2>
        <p className="text-[12.5px] text-gray-400 font-medium mt-0.5">{subtitle}</p>
      </div>
      <span className="bg-teal-50 text-teal-700 text-[11.5px] font-bold px-3 py-1 rounded-full font-mono">
        {count} / {total}
      </span>
    </div>
  )
}

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const allSlots = [...xraySlots, ...clinicalSlots]
  const [files, setFiles] = useState<FilesState>(() =>
    Object.fromEntries(allSlots.map(s => [s.id, null]))
  )
  const [now, setNow] = useState(new Date())

  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 30_000)
    return () => clearInterval(t)
  }, [])

  const handleUpload = (id: string, file: File) => {
    setFiles(prev => ({ ...prev, [id]: { file, preview: URL.createObjectURL(file) } }))
  }

  const handleRemove = (id: string) => {
    setFiles(prev => {
      const copy = { ...prev }
      if (copy[id]?.preview) URL.revokeObjectURL(copy[id]!.preview)
      copy[id] = null
      return copy
    })
  }

  const bulkInputRef = useRef<HTMLInputElement>(null)

  const handleBulkUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files || []).slice(0, allSlots.length)
    setFiles(prev => {
      const copy = { ...prev }
      selected.forEach((file, i) => {
        const slot = allSlots[i]
        if (copy[slot.id]?.preview) URL.revokeObjectURL(copy[slot.id]!.preview)
        copy[slot.id] = { file, preview: URL.createObjectURL(file) }
      })
      return copy
    })
    e.target.value = ''
  }

  const xrayCount = xraySlots.filter(s => files[s.id]).length
  const clinicalCount = clinicalSlots.filter(s => files[s.id]).length
  const totalCount = xrayCount + clinicalCount
  const totalSlots = allSlots.length
  const progressPct = (totalCount / totalSlots) * 100

  const hasXray = xrayCount > 0
  const hasClinical = clinicalCount > 0
  const diagnoseReady = hasXray && hasClinical
  const [showDashboard, setShowDashboard] = useState(false)

  const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })
  const dateStr = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })

  if (showDashboard) {
    return (
      <DiagnosisDashboard
        onBack={() => setShowDashboard(false)}
        uploadedImages={files}
      />
    )
  }

  return (
    <div className="min-h-screen bg-[#EEF3F7] pb-16" style={{ fontFamily: "'Sora', sans-serif" }}>

      {/* ─── Google Font import ─── */}
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');`}</style>

      {/* ─── Header ─── */}
      <header className="sticky top-0 z-50 shadow-lg" style={{ background: 'linear-gradient(135deg, #0D8A82 0%, #17B3A8 50%, #1EC8BC 100%)' }}>
        <div className="max-w-[1100px] mx-auto px-8 h-[68px] flex items-center justify-between">

          {/* Left: Logo + User */}
          <div className="flex items-center gap-4 shrink-0">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center border border-white/30 backdrop-blur-sm" style={{ background: 'rgba(255,255,255,0.18)' }}>
                <ToothLogoIcon />
              </div>
              <span className="text-xl font-extrabold text-white tracking-tight">Tooth</span>
            </div>
            <div className="w-px h-7 bg-white/25" />
            <div className="flex items-center gap-2 rounded-full py-1.5 pl-1.5 pr-4 border border-white/25 backdrop-blur-sm" style={{ background: 'rgba(255,255,255,0.15)' }}>
              <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white shrink-0"
                style={{ background: 'linear-gradient(135deg, #FFD166, #EF476F)' }}>N</div>
              <span className="text-[13px] font-semibold text-white">Nguyễn Văn An</span>
            </div>
          </div>

          {/* Center: Search */}
          <div className="flex-1 max-w-md mx-10 relative">
            <div className="absolute left-3.5 top-1/2 -translate-y-1/2"><SearchIcon /></div>
            <input
              type="text"
              placeholder="Tìm kiếm bệnh nhân hoặc mã hồ sơ..."
              className="w-full pl-10 pr-4 py-2 rounded-full text-[13px] outline-none transition-all"
              style={{
                fontFamily: "'Sora', sans-serif",
                background: 'rgba(255,255,255,0.18)',
                border: '1.5px solid rgba(255,255,255,0.3)',
                color: 'white',
              }}
            />
          </div>

          {/* Right: Clock + Bell */}
          <div className="flex items-center gap-4 shrink-0">
            <div className="text-right">
              <div className="text-[14px] font-bold text-white" style={{ fontFamily: "'DM Mono', monospace" }}>{timeStr}</div>
              <div className="text-[11px] text-white/75 mt-0.5">{dateStr}</div>
            </div>
            <div className="w-9 h-9 rounded-xl flex items-center justify-center border border-white/25 backdrop-blur-sm cursor-pointer hover:bg-white/25 transition-colors"
              style={{ background: 'rgba(255,255,255,0.15)' }}>
              <BellIcon />
            </div>
          </div>
        </div>
      </header>

      {/* ─── Main ─── */}
      <main className="max-w-[1100px] mx-auto px-8 py-9">

        {/* Page Header */}
        <div className="flex items-start justify-between mb-7">
          <div>
            <div className="flex items-center gap-3 mb-1.5">
              <h1 className="text-[24px] font-extrabold text-[#0D1B2A] tracking-tight">
                Dữ liệu Chẩn đoán Hình ảnh
              </h1>
              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[12px] font-bold
                ${diagnoseReady ? 'bg-emerald-50 text-emerald-600' : 'bg-gray-100 text-gray-400'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${diagnoseReady ? 'bg-emerald-400' : 'bg-gray-300'}`} />
                {diagnoseReady ? 'Sẵn sàng chẩn đoán' : 'Chưa đủ dữ liệu'}
              </span>
            </div>
            <p className="text-[13.5px] text-gray-400 font-medium">
              Tải lên đầy đủ phim X-quang &amp; ảnh lâm sàng để kích hoạt chẩn đoán đa phương thức.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Bulk Upload Button */}
            <button
              onClick={() => bulkInputRef.current?.click()}
              className="flex items-center gap-2 px-5 py-3 rounded-[14px] text-[14px] font-bold border-[1.5px] border-teal-300 text-teal-700 bg-teal-50 hover:bg-teal-100 hover:-translate-y-0.5 transition-all duration-200"
            >
              <BulkUploadIcon />
              Tải lên tất cả
              <span className="text-[11px] font-semibold bg-teal-200 text-teal-800 px-2 py-0.5 rounded-full">4 ảnh</span>
            </button>
            <input
              ref={bulkInputRef}
              type="file"
              accept="image/*"
              multiple
              className="hidden"
              onChange={handleBulkUpload}
            />

            {/* Diagnose Button */}
            <button
              disabled={!diagnoseReady}
              onClick={() => diagnoseReady && setShowDashboard(true)}
              className={`flex items-center gap-2.5 px-7 py-3 rounded-[14px] text-[14px] font-bold transition-all duration-200
                ${diagnoseReady
                  ? 'text-white shadow-lg hover:-translate-y-0.5 hover:shadow-xl active:translate-y-0'
                  : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                }`}
              style={diagnoseReady ? { background: 'linear-gradient(135deg, #0096C7, #2EC4B6)', boxShadow: '0 8px 24px rgba(0,150,199,0.35)' } : {}}
            >
              <DiagnoseIcon />
              Chẩn đoán ngay
            </button>
          </div>
        </div>

        {/* Progress Strip */}
        <div className="bg-white rounded-2xl border border-gray-100 shadow-sm px-6 py-4 mb-8 flex items-center gap-5">
          <span className="text-[13px] font-semibold text-gray-500 whitespace-nowrap">Tiến độ tải lên</span>
          <div className="flex-1 h-2 rounded-full bg-gray-100 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${progressPct}%`,
                background: 'linear-gradient(90deg, #2EC4B6, #0096C7)'
              }}
            />
          </div>
          <span className="text-[13px] font-bold text-teal-600 whitespace-nowrap" style={{ fontFamily: "'DM Mono', monospace" }}>
            {totalCount} / {totalSlots}
          </span>
        </div>

        {/* X-Ray Section */}
        <div className="mb-8">
          <SectionHeader title="Radiographic Imaging" subtitle="Phim X-quang kỹ thuật số" count={xrayCount} total={xraySlots.length} />
          <div className="flex flex-wrap gap-4">
            {xraySlots.map(slot => (
              <ImageSlot key={slot.id} slot={slot} uploaded={files[slot.id]} onUpload={handleUpload} onRemove={handleRemove} />
            ))}
          </div>
        </div>

        {/* Clinical Section */}
        <div className="mb-8">
          <SectionHeader title="Clinical Imaging Data" subtitle="Ảnh lâm sàng thực tế" count={clinicalCount} total={clinicalSlots.length} />
          <div className="flex flex-wrap gap-4">
            {clinicalSlots.map(slot => (
              <ImageSlot key={slot.id} slot={slot} uploaded={files[slot.id]} onUpload={handleUpload} onRemove={handleRemove} />
            ))}
          </div>
        </div>

      </main>
    </div>
  )
}