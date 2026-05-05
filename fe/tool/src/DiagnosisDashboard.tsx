import { useState, useCallback, useEffect } from 'react'
import axios from 'axios'

// ══════════════════════════════════════════════════════════════════════════════
// TYPES  (mirror schemas/diagnosis_schema.py)
// ══════════════════════════════════════════════════════════════════════════════
type MStatus = 'normal' | 'warning' | 'danger'

interface ReportSummary {
  total_teeth: number
  healthy: number
  cavities: number
  needs_treatment: number
}

interface IssueItem {
  issue_name: string
  confidence: number
  issue_bbox: [number, number, number, number]
  status: 'NEEDS_TREATMENT' | 'ALREADY_TREATED'
}

interface ToothAnalysis {
  tooth_number: string
  tooth_bbox: [number, number, number, number]
  issues: IssueItem[]
}

interface LandmarkPoint {
  name: string
  x: number
  y: number
}

interface CephMetrics {
  SNA: number
  SNB: number
  ANB: number
}

interface CephAnalysis {
  landmarks: LandmarkPoint[]
  metrics: CephMetrics
  conclusion: string
  conclusion_detail: string
}

interface LrcAnalysis {
  angles: { SNA: number; SNB: number; ANB: number }
  diagnosis: string
}

interface CephAiAnalysis {
  skeletal_summary: string
  sna_interpretation: string
  snb_interpretation: string
  anb_interpretation: string
  clinical_implications: string[]
  treatment_plan: string[]
  severity: 'low' | 'medium' | 'high'
}

interface ConsultationIssue {
  issue: string
  detail: string
  recommendation: string
}

interface Consultation {
  overall_assessment: string[]
  main_issues: ConsultationIssue[]
}

interface ToothDetail {
  tooth_number: string
  disease_name: string
  latin_name: string
  treatment_method: string
  estimated_duration: string
  severity_percent: number
  status: 'NEEDS_TREATMENT' | 'ALREADY_TREATED' | 'INFO'
}

interface FullReportResponse {
  summary: ReportSummary
  panoramic_analysis: ToothAnalysis[]
  ceph_analysis: CephAnalysis | null
  ceph_ai_analysis: CephAiAnalysis | null
  consultation: Consultation
  teeth_details: ToothDetail[]
}

// ══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ══════════════════════════════════════════════════════════════════════════════
const API_URL        = 'http://127.0.0.1:8000/api/v1/diagnosis/full-report'
const LRC_IMAGE_API  = 'http://127.0.0.1:8000/api/predict_analysis_image'
const LRC_DATA_API   = 'http://127.0.0.1:8000/api/predict_analysis_data'
const CEPH_AI_API    = 'http://127.0.0.1:8000/api/ceph_ai_analysis'

const CEPH_NORMS = {
  SNA: { label: '82 ± 2°', min: 80, max: 84, display: 'SNA' },
  SNB: { label: '80 ± 2°', min: 78, max: 82, display: 'SNB' },
  ANB: { label: '2 ± 2°',  min: 0,  max: 4,  display: 'ANB' },
} as const

// ══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════════════════════
const mStatusColor = (s: MStatus) =>
  s === 'normal' ? '#10B981' : s === 'warning' ? '#F59E0B' : '#EF4444'
const mStatusBg = (s: MStatus) =>
  s === 'normal' ? '#F0FDF4' : s === 'warning' ? '#FFFBEB' : '#FEF2F2'
const mStatusLabel = (s: MStatus) =>
  s === 'normal' ? 'Bình thường' : s === 'warning' ? 'Chú ý' : 'Bất thường'

function getLrcConclusion(diagnosis: string, anb: number): { conclusion: string; conclusion_detail: string } {
  if (diagnosis === 'Hô') return {
    conclusion: 'Khớp cắn Loại II (Class II) — Hô',
    conclusion_detail: `ANB = ${anb.toFixed(2)}° > 4° — Xương hàm trên nhô ra trước so với hàm dưới`,
  }
  if (diagnosis === 'Móm') return {
    conclusion: 'Khớp cắn Loại III (Class III) — Móm',
    conclusion_detail: `ANB = ${anb.toFixed(2)}° < 0° — Xương hàm dưới nhô ra trước so với hàm trên`,
  }
  return {
    conclusion: 'Khớp cắn Loại I (Class I)',
    conclusion_detail: `ANB = ${anb.toFixed(2)}° — Không có dấu hiệu hô hoặc móm đáng kể`,
  }
}

function classifyMetric(key: keyof typeof CEPH_NORMS, value: number): MStatus {
  const { min, max } = CEPH_NORMS[key]
  if (value >= min && value <= max) return 'normal'
  const range = max - min
  if (value >= min - range && value <= max + range) return 'warning'
  return 'danger'
}

function severityToPriority(pct: number) {
  if (pct >= 75) return { label: 'Cao',             bg: '#EF4444' }
  if (pct >= 50) return { label: 'Trung bình', bg: '#F59E0B' }
  return             { label: 'Thấp',           bg: '#10B981' }
}

// ══════════════════════════════════════════════════════════════════════════════
// ICONS (inline SVG)
// ══════════════════════════════════════════════════════════════════════════════
function ToothIcon({ size = 20, color = 'currentColor' }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
      <path d="M12 2C9 2 6 4.5 6 8c0 2.5 1 3.8 1.8 5.2C8.5 14.7 8.8 16 9 17.5c.2 1.2.8 2.5 2 2.5h2c1.2 0 1.8-1.3 2-2.5.2-1.5.5-2.8 1.2-4.3C17 11.8 18 10.5 18 8c0-3.5-3-6-6-6z" />
    </svg>
  )
}
function HeartIcon({ size = 18, color = 'currentColor' }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
    </svg>
  )
}
function WarningIcon({ size = 18, color = 'currentColor' }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  )
}
function ZoomInIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
      <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
      <line x1="11" y1="8" x2="11" y2="14" /><line x1="8" y1="11" x2="14" y2="11" />
    </svg>
  )
}
function ZoomOutIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
      <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
      <line x1="8" y1="11" x2="14" y2="11" />
    </svg>
  )
}
function BrainIcon({ size = 16, color = 'white' }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.8">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.87-3.73A3 3 0 0 1 5 12a3 3 0 0 1 .64-1.87 2.5 2.5 0 0 1 1.87-3.73A2.5 2.5 0 0 1 9.5 2Z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 1.87-3.73A3 3 0 0 0 19 12a3 3 0 0 0-.64-1.87 2.5 2.5 0 0 0-1.87-3.73A2.5 2.5 0 0 0 14.5 2Z" />
    </svg>
  )
}
function RulerIcon({ size = 15, color = 'white' }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
      <path d="M21.3 8.7 8.7 21.3c-1 1-2.5 1-3.4 0l-2.6-2.6c-1-1-1-2.5 0-3.4L15.3 2.7c1-1 2.5-1 3.4 0l2.6 2.6c1 1 1 2.5 0 3.4Z" />
      <path d="m7.5 10.5 2 2" /><path d="m10.5 7.5 2 2" /><path d="m13.5 4.5 2 2" /><path d="m4.5 13.5 2 2" />
    </svg>
  )
}
function ClockIcon({ size = 13, color = '#9CA3AF' }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
      <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
    </svg>
  )
}
function ImageIcon({ size = 36, color = '#CBD5E1' }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.4">
      <rect x="3" y="3" width="18" height="18" rx="2.5" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </svg>
  )
}
function BackIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  )
}
// ══════════════════════════════════════════════════════════════════════════════
// SUB-COMPONENTS
// ══════════════════════════════════════════════════════════════════════════════

function BlockHeader({ icon, title, right }: { icon: React.ReactNode; title: string; right?: React.ReactNode }) {
  return (
    <div
      className="flex items-center gap-2.5 px-5 py-3 shrink-0"
      style={{ background: 'linear-gradient(135deg, #0D9488 0%, #14B8A6 100%)' }}
    >
      {icon}
      <span className="text-[13px] font-bold text-white uppercase tracking-wide flex-1">{title}</span>
      {right}
    </div>
  )
}

function SummaryCard({
  label, value, sub, icon, valueColor = '#1F2937', bg = '#FFFFFF', border = '#E5E7EB',
}: {
  label: string; value: number; sub?: string; icon: React.ReactNode
  valueColor?: string; bg?: string; border?: string
}) {
  return (
    <div
      className="rounded-xl shadow-sm px-5 py-4 flex flex-col gap-1.5"
      style={{ background: bg, border: `1.5px solid ${border}` }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[11.5px] font-semibold text-gray-500">{label}</span>
        {icon}
      </div>
      <span
        className="text-[34px] font-extrabold leading-none"
        style={{ color: valueColor, fontFamily: "'DM Mono', monospace" }}
      >{value}</span>
      {sub && <span className="text-[10.5px] text-gray-400">{sub}</span>}
    </div>
  )
}

function ClinicalCard({ label, preview }: { label: string; preview?: string | null }) {
  return (
    <div className="flex-1 flex flex-col bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm min-h-0">
      <div className="px-3 py-2 border-b border-gray-100 bg-gray-50 shrink-0">
        <p className="text-[11.5px] font-semibold text-gray-600">{label}</p>
      </div>
      <div className="flex-1 flex items-center justify-center bg-slate-50">
        {preview ? (
          <img src={preview} alt={label} className="w-full h-full object-cover" />
        ) : (
          <div className="flex flex-col items-center gap-2 py-6">
            <ImageIcon />
            <p className="text-[10px] text-gray-400">Chưa có ảnh</p>
          </div>
        )}
      </div>
    </div>
  )
}

function LoadingOverlay() {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(15,23,42,0.72)', backdropFilter: 'blur(6px)' }}
    >
      <div className="bg-white rounded-2xl px-12 py-10 flex flex-col items-center gap-5 shadow-2xl">
        <div className="relative w-20 h-20">
          <div className="absolute inset-0 rounded-full border-[5px] border-teal-100" />
          <div className="absolute inset-0 rounded-full border-[5px] border-teal-500 border-t-transparent animate-spin" />
          <div className="absolute inset-4 flex items-center justify-center">
            <BrainIcon size={26} color="#0D9488" />
          </div>
        </div>
        <div className="text-center">
          <p className="text-[16px] font-bold text-gray-800 mb-1">AI đang phân tích...</p>
          <p className="text-[12.5px] text-gray-400">Đang xử lý X-quang Panoramic và Cephalometric</p>
        </div>
        <div className="flex gap-2">
          {[0, 1, 2].map(i => (
            <div
              key={i}
              className="w-2.5 h-2.5 rounded-full bg-teal-400 animate-bounce"
              style={{ animationDelay: `${i * 160}ms` }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function EmptyState({ onBack }: { onBack?: () => void }) {
  return (
    <div className="min-h-screen bg-[#F3F4F6] flex items-center justify-center" style={{ fontFamily: "'Sora', sans-serif" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');`}</style>
      <div className="flex flex-col items-center gap-6 text-center max-w-md px-6">
        <div
          className="w-24 h-24 rounded-full flex items-center justify-center"
          style={{ background: 'linear-gradient(135deg, #CCFBF1, #99F6E4)' }}
        >
          <ToothIcon size={44} color="#0D9488" />
        </div>
        <div>
          <h2 className="text-[22px] font-extrabold text-gray-800 mb-2">Chưa có dữ liệu hình ảnh</h2>
          <p className="text-[13.5px] text-gray-500 leading-relaxed">
            Vui lòng tải lên ít nhất ảnh X-quang Panoramic để bắt đầu chẩn đoán bằng AI.
          </p>
        </div>
        {onBack && (
          <button
            onClick={onBack}
            className="flex items-center gap-2 px-6 py-3 rounded-xl text-[13.5px] font-bold text-white shadow-md transition-all hover:-translate-y-0.5"
            style={{ background: 'linear-gradient(135deg, #0D9488, #14B8A6)' }}
          >
            <BackIcon />
            Quay lại tải ảnh
          </button>
        )}
      </div>
    </div>
  )
}

function ToothBoundingBox({ tooth, naturalW, naturalH }: {
  tooth: ToothAnalysis; naturalW: number; naturalH: number
}) {
  const [x1, y1, x2, y2] = tooth.tooth_bbox
  const needsTreatment = tooth.issues.some(i => i.status === 'NEEDS_TREATMENT')
  const toothColor = needsTreatment ? '#EF4444' : '#F59E0B'

  return (
    <>
      <div
        className="absolute pointer-events-none"
        style={{
          left:   `${(x1 / naturalW) * 100}%`,
          top:    `${(y1 / naturalH) * 100}%`,
          width:  `${((x2 - x1) / naturalW) * 100}%`,
          height: `${((y2 - y1) / naturalH) * 100}%`,
        }}
      >
        <div
          className="w-full h-full rounded"
          style={{ border: `2px dashed ${toothColor}`, boxShadow: `0 0 10px ${toothColor}55` }}
        />
        <span
          className="absolute left-1/2 -translate-x-1/2 text-[9px] font-bold px-1.5 py-[2px] rounded-sm whitespace-nowrap"
          style={{ bottom: '-18px', background: toothColor, color: '#fff' }}
        >
          {tooth.tooth_number}
        </span>
      </div>

    </>
  )
}

// Lines to draw on the ceph X-ray (matching reference image style)
const CEPH_LINES: Array<{
  pts: string[]          // landmark names in order
  color: string
  dash?: string
  opacity?: number
}> = [
  { pts: ['S', 'N'],          color: '#22C55E', dash: '8,5' },  // SN plane (dashed)
  { pts: ['N', 'A'],          color: '#22C55E' },                // NA line
  { pts: ['N', 'B'],          color: '#22C55E' },                // NB line
  { pts: ['A', 'B'],          color: '#22C55E' },                // AB line
  { pts: ['Go', 'Me'],        color: '#A855F7' },                // Mandibular plane (Go-Me)
  { pts: ['Me', 'Pog'],       color: '#A855F7' },                // Mandibular plane (Me-Pog)
  { pts: ['Po', 'Or'],        color: '#F59E0B', dash: '6,4' },  // Frankfort horizontal
  { pts: ['ANS', 'PNS'],      color: '#3B82F6', dash: '6,4' },  // Palatal plane
  { pts: ['UIE', 'UIA'],      color: '#F97316' },               // Upper incisor axis
  { pts: ['LIE', 'LIA'],      color: '#FB923C' },               // Lower incisor axis
  { pts: ['Ar', 'Go'],        color: '#EC4899' },                // Ramus
  { pts: ['S', 'Ar'],         color: '#6366F1', dash: '5,4' },  // S-Ar line
  { pts: ['N', 'Pog'],        color: '#14B8A6', dash: '5,4' },  // Facial line (N-Pog)
]

function CephLines({ landmarks, naturalW, naturalH }: {
  landmarks: LandmarkPoint[]
  naturalW: number
  naturalH: number
}) {
  const lmMap: Record<string, LandmarkPoint> = {}
  for (const lm of landmarks) lmMap[lm.name] = lm

  const strokeW = Math.max(1.5, naturalW / 320)

  return (
    <svg
      viewBox={`0 0 ${naturalW} ${naturalH}`}
      className="absolute inset-0 w-full h-full pointer-events-none"
      style={{ top: 0, left: 0 }}
    >
      {CEPH_LINES.map((seg, si) =>
        seg.pts.slice(0, -1).map((fromName, i) => {
          const toName = seg.pts[i + 1]
          const from = lmMap[fromName]
          const to   = lmMap[toName]
          if (!from || !to) return null
          return (
            <line
              key={`${si}-${i}`}
              x1={from.x} y1={from.y}
              x2={to.x}   y2={to.y}
              stroke={seg.color}
              strokeWidth={strokeW}
              strokeDasharray={seg.dash}
              strokeLinecap="round"
              opacity={seg.opacity ?? 0.9}
            />
          )
        })
      )}
    </svg>
  )
}

// Landmark full names (reference chart)
const LM_INFO: Record<string, { full: string; desc: string }> = {
  S:    { full: 'Sella',                     desc: 'Trung tâm hố yên, nằm ở giữa hố yên (vùng tuyến yên)' },
  N:    { full: 'Nasion',                    desc: 'Giao điểm giữa xương mũi và xương trán – đỉnh sống mũi trên' },
  Or:   { full: 'Orbitale',                  desc: 'Điểm thấp nhất ở bờ dưới của mắt' },
  Po:   { full: 'Porion',                    desc: 'Điểm cao nhất của lỗ tai ngoài – dùng xác định mặt phẳng Frankfort' },
  A:    { full: 'Subspinale (Point A)',       desc: 'Điểm lõm nhất trên xương hàm trên (giữa mũi và răng trên)' },
  ANS:  { full: 'Anterior Nasal Spine',      desc: 'Gai mũi trước – điểm trước nhất của xương khẩu cái' },
  PNS:  { full: 'Posterior Nasal Spine',     desc: 'Gai mũi sau – điểm sau cùng của xương khẩu cái' },
  B:    { full: 'Supramentale (Point B)',     desc: 'Điểm lõm nhất trên xương hàm dưới (giữa môi dưới và cằm)' },
  Pog:  { full: 'Pogonion',                  desc: 'Điểm nhô nhất ở cằm xương' },
  Gn:   { full: 'Gnathion',                  desc: 'Điểm ở giữa Me và Pog, thường là đáy cằm' },
  Me:   { full: 'Menton',                    desc: 'Điểm thấp nhất của xương cằm' },
  Go:   { full: 'Gonion',                    desc: 'Góc hàm dưới – điểm giao nhau của cạnh sau và dưới xương hàm dưới' },
  Ar:   { full: 'Articulare',                desc: 'Giao điểm bề mặt sau lồi cầu với nền sọ' },
  Ba:   { full: 'Basion',                    desc: 'Điểm thấp nhất của xương chẩm' },
  Pt:   { full: 'Pterygomaxillary (PtM)',    desc: 'Vùng khe giữa xương cánh bướm và hàm trên' },
  CF:   { full: 'Center of Face',            desc: 'Trung tâm khuôn mặt' },
  UIE:  { full: 'Upper Incisor Edge',        desc: 'Rìa cắn răng cửa trên' },
  LIE:  { full: 'Lower Incisor Edge',        desc: 'Rìa cắn răng cửa dưới' },
  UIA:  { full: 'Upper Incisor Apex',        desc: 'Chóp chân răng cửa trên' },
  LIA:  { full: 'Lower Incisor Apex',        desc: 'Chóp chân răng cửa dưới' },
  UL:   { full: 'Upper Lip',                 desc: 'Môi trên' },
  LL:   { full: 'Lower Lip',                 desc: 'Môi dưới' },
  Stms: { full: 'Stomion Superius',          desc: 'Điểm giữa bờ dưới của môi trên' },
  Stmi: { full: 'Stomion Inferius',          desc: 'Điểm giữa bờ trên của môi dưới' },
  Pg:   { full: "Soft-tissue Pogonion (Pg')", desc: 'Phiên bản mô mềm của Pogonion' },
  Dt:   { full: 'Dtubercle',                 desc: 'Mấu cộng' },
  Xi:   { full: 'Xi Point',                  desc: 'Điểm trung tâm của cành đứng xương hàm dưới' },
  Pm:   { full: 'Protuberance Menti',        desc: 'Lồi cằm' },
  Na:   { full: 'Nasal Tip',                 desc: 'Đỉnh mũi mô mềm' },
}

// Categorize landmarks by type for color-coding
const LANDMARK_COLORS: Record<string, string> = {
  // Cranial base (green)
  S: '#22C55E', N: '#22C55E', Ba: '#22C55E', Ar: '#22C55E',
  // Maxillary skeletal (blue)
  A: '#3B82F6', ANS: '#3B82F6', PNS: '#3B82F6',
  // Mandibular skeletal (purple)
  B: '#A855F7', Pog: '#A855F7', Gn: '#A855F7', Me: '#A855F7', Go: '#A855F7', Xi: '#A855F7', Pm: '#A855F7',
  // Frankfort plane (orange/yellow)
  Or: '#F59E0B', Po: '#F59E0B',
  // Dental (orange)
  UIE: '#F97316', UIA: '#F97316', LIE: '#FB923C', LIA: '#FB923C',
  // Soft tissue (pink/red)
  UL: '#EC4899', LL: '#EC4899', Stms: '#EC4899', Stmi: '#EC4899', Na: '#EC4899', Pg: '#EC4899', Dt: '#14B8A6',
  // Reference points (cyan)
  Pt: '#14B8A6', CF: '#14B8A6',
}

function LandmarkDot({ point, naturalW, naturalH }: {
  point: LandmarkPoint; naturalW: number; naturalH: number
}) {
  const info = LM_INFO[point.name]
  const dotColor = LANDMARK_COLORS[point.name] ?? '#EF4444'
  return (
    <div
      className="absolute group"
      style={{
        left:      `${(point.x / naturalW) * 100}%`,
        top:       `${(point.y / naturalH) * 100}%`,
        transform: 'translate(-50%, -50%)',
        zIndex: 10,
      }}
    >
      {/* dot */}
      <div className="w-2.5 h-2.5 rounded-full border-2 border-white/90 shadow-md cursor-pointer"
        style={{ background: dotColor, boxShadow: '0 0 0 1px rgba(0,0,0,0.4)' }} />

      {/* label */}
      <span
        className="absolute whitespace-nowrap text-[7.5px] font-bold pointer-events-none select-none"
        style={{
          left: '10px', top: '-4px',
          fontFamily: "'DM Mono', monospace",
          color: dotColor,
          textShadow: '0 1px 3px rgba(0,0,0,0.9)',
        }}
      >
        {point.name}
      </span>

      {/* tooltip on hover */}
      {info && (
        <div
          className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2
                     bg-gray-900/95 text-white rounded-lg px-3 py-2 shadow-xl
                     opacity-0 group-hover:opacity-100 transition-opacity duration-150
                     pointer-events-none select-none"
          style={{ minWidth: 180, maxWidth: 260, zIndex: 50 }}
        >
          <p className="text-[11px] font-bold mb-0.5"
            style={{ fontFamily: "'DM Mono', monospace", color: dotColor }}>
            {point.name} — {info.full}
          </p>
          <p className="text-[10px] text-gray-300 leading-snug">{info.desc}</p>
        </div>
      )}
    </div>
  )
}

const TOOTH_CARD_THEME = {
  NEEDS_TREATMENT: {
    bg: 'bg-red-50', border: 'border-red-200',
    iconBg: 'bg-red-100', iconColor: '#EF4444',
    innerBorder: 'border-red-100', textColor: 'text-red-700',
    barBg: 'bg-red-100', barGradient: 'linear-gradient(90deg, #F87171, #DC2626)',
    barText: 'text-red-500', severityLabel: 'Mức độ nghiêm trọng',
  },
  ALREADY_TREATED: {
    bg: 'bg-green-50', border: 'border-green-200',
    iconBg: 'bg-green-100', iconColor: '#10B981',
    innerBorder: 'border-green-100', textColor: 'text-green-700',
    barBg: 'bg-green-100', barGradient: 'linear-gradient(90deg, #6EE7B7, #059669)',
    barText: 'text-green-600', severityLabel: 'Mức độ can thiệp',
  },
  INFO: {
    bg: 'bg-blue-50', border: 'border-blue-200',
    iconBg: 'bg-blue-100', iconColor: '#3B82F6',
    innerBorder: 'border-blue-100', textColor: 'text-blue-700',
    barBg: 'bg-blue-100', barGradient: 'linear-gradient(90deg, #93C5FD, #2563EB)',
    barText: 'text-blue-500', severityLabel: 'Mức liên quan',
  },
} as const

function ToothCard({ tooth }: { tooth: ToothDetail }) {
  const status = tooth.status ?? 'NEEDS_TREATMENT'
  const theme = TOOTH_CARD_THEME[status] ?? TOOTH_CARD_THEME.NEEDS_TREATMENT
  const priority = severityToPriority(tooth.severity_percent)
  return (
    <div className={`${theme.bg} border ${theme.border} rounded-xl p-4 flex flex-col gap-3`}>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2.5">
          <div className={`w-9 h-9 rounded-full ${theme.iconBg} flex items-center justify-center shrink-0`}>
            <ToothIcon size={17} color={theme.iconColor} />
          </div>
          <div>
            <p className="text-[13.5px] font-bold text-gray-800">Răng {tooth.tooth_number}</p>
            {tooth.latin_name && (
              <p className="text-[11px] text-gray-500 italic">{tooth.latin_name}</p>
            )}
          </div>
        </div>
        <span
          className="text-[10px] font-bold px-2.5 py-0.5 rounded-full shrink-0 text-white"
          style={{ background: priority.bg }}
        >
          {priority.label}
        </span>
      </div>
      <p className="text-[12px] text-gray-700 font-medium leading-snug">{tooth.disease_name}</p>
      <div className={`bg-white border ${theme.innerBorder} rounded-lg px-3 py-1.5`}>
        <p className={`text-[11.5px] font-semibold ${theme.textColor}`}>{tooth.treatment_method}</p>
      </div>
      <div className="flex items-center gap-1.5 text-[11px] text-gray-500">
        <ClockIcon />{tooth.estimated_duration}
      </div>
      <div>
        <div className="flex justify-between text-[10.5px] font-semibold mb-1">
          <span className="text-gray-400">{theme.severityLabel}</span>
          <span className={theme.barText}>{tooth.severity_percent}%</span>
        </div>
        <div className={`h-[6px] w-full rounded-full ${theme.barBg} overflow-hidden`}>
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{
              width: `${tooth.severity_percent}%`,
              background: theme.barGradient,
            }}
          />
        </div>
      </div>
    </div>
  )
}

function SparkleIcon({ size = 16, color = 'white' }: { size?: number; color?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill={color}>
      <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z" />
    </svg>
  )
}

const SEVERITY_CONFIG = {
  low:    { label: 'Nhẹ',         bg: '#F0FDF4', border: '#6EE7B7', color: '#10B981', dot: '#10B981' },
  medium: { label: 'Trung bình',  bg: '#FFFBEB', border: '#FDE68A', color: '#F59E0B', dot: '#F59E0B' },
  high:   { label: 'Nghiêm trọng',bg: '#FEF2F2', border: '#FECACA', color: '#EF4444', dot: '#EF4444' },
} as const

function CephAiCard({ analysis }: { analysis: CephAiAnalysis }) {
  const sev = SEVERITY_CONFIG[analysis.severity]
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
      <BlockHeader
        icon={<SparkleIcon size={15} />}
        title="Phân Tích Cephalometric AI Chuyên Sâu"
        right={
          <span
            className="text-[11px] font-bold px-3 py-0.5 rounded-full"
            style={{ background: sev.bg, color: sev.color, border: `1px solid ${sev.border}` }}
          >
            Mức độ: {sev.label}
          </span>
        }
      />
      <div className="p-6 flex flex-col gap-5">

        {/* Tóm tắt */}
        <div
          className="rounded-xl px-4 py-3 border"
          style={{ background: 'linear-gradient(135deg,#F0FDFA,#E6FFFA)', borderColor: '#99F6E4' }}
        >
          <p className="text-[10.5px] font-bold uppercase tracking-widest text-teal-600 mb-1">Tóm tắt</p>
          <p className="text-[13.5px] font-semibold text-gray-800 leading-relaxed">{analysis.skeletal_summary}</p>
        </div>

        {/* 3 góc phân tích */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[
            { key: 'SNA', label: 'Góc SNA — Vị trí Hàm Trên', text: analysis.sna_interpretation, color: '#3B82F6', bg: '#EFF6FF', border: '#BFDBFE' },
            { key: 'SNB', label: 'Góc SNB — Vị trí Hàm Dưới', text: analysis.snb_interpretation, color: '#A855F7', bg: '#FAF5FF', border: '#E9D5FF' },
            { key: 'ANB', label: 'Góc ANB — Quan hệ Sagittal', text: analysis.anb_interpretation, color: '#F59E0B', bg: '#FFFBEB', border: '#FDE68A' },
          ].map(item => (
            <div
              key={item.key}
              className="rounded-xl px-4 py-3 border flex flex-col gap-1.5"
              style={{ background: item.bg, borderColor: item.border }}
            >
              <div className="flex items-center gap-2">
                <span
                  className="text-[10px] font-black px-2 py-0.5 rounded"
                  style={{ background: item.color, color: '#fff', fontFamily: "'DM Mono', monospace" }}
                >
                  {item.key}
                </span>
                <p className="text-[10.5px] font-bold uppercase tracking-wide" style={{ color: item.color }}>
                  {item.label.split('—')[1]?.trim()}
                </p>
              </div>
              <p className="text-[12px] text-gray-700 leading-relaxed">{item.text}</p>
            </div>
          ))}
        </div>

        {/* Hậu quả lâm sàng + Kế hoạch điều trị */}
        <div className="flex flex-wrap gap-6">
          {analysis.clinical_implications.length > 0 && (
            <div className="flex-1 min-w-[220px]">
              <p className="text-[10.5px] font-bold text-amber-700 uppercase tracking-widest mb-3">
                Hậu quả lâm sàng
              </p>
              <ul className="flex flex-col gap-2">
                {analysis.clinical_implications.map((item, i) => (
                  <li key={i} className="flex gap-2.5 text-[12px] text-gray-700 leading-relaxed">
                    <span
                      className="w-1.5 h-1.5 rounded-full mt-[6px] shrink-0"
                      style={{ background: sev.dot }}
                    />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          )}
          {analysis.clinical_implications.length > 0 && analysis.treatment_plan.length > 0 && (
            <div className="w-px bg-gray-100 self-stretch hidden md:block" />
          )}
          {analysis.treatment_plan.length > 0 && (
            <div className="flex-1 min-w-[220px]">
              <p className="text-[10.5px] font-bold text-teal-700 uppercase tracking-widest mb-3">
                Kế hoạch điều trị đề xuất
              </p>
              <ol className="flex flex-col gap-2">
                {analysis.treatment_plan.map((step, i) => (
                  <li key={i} className="flex gap-2.5 text-[12px] text-gray-700 leading-relaxed">
                    <span
                      className="w-5 h-5 rounded-full flex items-center justify-center text-white text-[9px] font-bold shrink-0 mt-[1px]"
                      style={{ background: 'linear-gradient(135deg, #0D9488, #14B8A6)', minWidth: 20 }}
                    >
                      {i + 1}
                    </span>
                    {step}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}

function CephalometricPlaceholder() {  const lm: Record<string, { x: number; y: number }> = {
    S:  { x: 148, y:  92 }, N:  { x: 210, y:  80 },
    A:  { x: 222, y: 196 }, B:  { x: 214, y: 232 },
    Pg: { x: 218, y: 255 }, Me: { x: 210, y: 274 },
    Go: { x: 105, y: 260 },
  }
  const lines = [
    { x1: lm.S.x-28, y1: lm.S.y+3,   x2: lm.N.x+18, y2: lm.N.y-2,  color: '#10B981', dash: '6,3' },
    { x1: lm.N.x,    y1: lm.N.y,     x2: lm.A.x,    y2: lm.A.y,    color: '#10B981', dash: '6,3' },
    { x1: lm.N.x,    y1: lm.N.y,     x2: lm.B.x,    y2: lm.B.y,    color: '#3B82F6', dash: '6,3' },
    { x1: lm.A.x,    y1: lm.A.y,     x2: lm.B.x,    y2: lm.B.y,    color: '#F59E0B', dash: ''    },
    { x1: lm.Go.x,   y1: lm.Go.y,    x2: lm.Me.x+28,y2: lm.Me.y+4, color: '#A855F7', dash: '6,3' },
  ]
  return (
    <svg viewBox="0 0 300 360" width="100%" height="100%" style={{ display: 'block' }}>
      <defs>
        <radialGradient id="ceph-bg" cx="55%" cy="38%" r="58%">
          <stop offset="0%" stopColor="#1E293B" /><stop offset="100%" stopColor="#0F172A" />
        </radialGradient>
      </defs>
      <rect width="300" height="360" fill="url(#ceph-bg)" />
      <path d="M 70,132 Q 72,96 80,76 Q 92,50 120,34 Q 150,16 188,14 Q 230,12 258,34 Q 280,52 284,84 Q 288,114 274,138 Q 260,158 242,168 Q 228,174 216,176"
        fill="none" stroke="#4B5563" strokeWidth="2.5" strokeLinecap="round" />
      <path d="M 70,132 Q 62,152 66,174 Q 72,194 90,206 Q 106,216 118,218"
        fill="none" stroke="#4B5563" strokeWidth="2.5" strokeLinecap="round" />
      <path d="M 105,260 Q 120,280 150,282 Q 178,284 198,272 Q 212,264 216,256 Q 220,248 218,240"
        fill="none" stroke="#4B5563" strokeWidth="2.5" strokeLinecap="round" />
      <path d="M 118,218 Q 122,234 120,248 Q 118,262 108,268 Q 105,270 105,260"
        fill="none" stroke="#374151" strokeWidth="2" strokeLinecap="round" />
      <ellipse cx="128" cy="136" rx="11" ry="6" fill="none" stroke="#4B5563" strokeWidth="1.5"
        transform="rotate(-18 128 136)" />
      <ellipse cx="238" cy="116" rx="23" ry="17" fill="none" stroke="#374151" strokeWidth="1.5" opacity="0.55" />
      <path d="M 210,80 Q 230,90 240,106 Q 246,116 240,124 Q 234,130 228,132 Q 222,132 218,140 Q 214,148 218,160 Q 220,170 218,178"
        fill="none" stroke="#4B5563" strokeWidth="2" strokeLinecap="round" />
      {lines.map((l, i) => (
        <line key={i} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2}
          stroke={l.color} strokeWidth="1.6"
          strokeDasharray={l.dash || undefined} opacity="0.88" />
      ))}
      {(Object.entries(lm) as [string, { x: number; y: number }][]).map(([key, p]) => (
        <g key={key}>
          <circle cx={p.x} cy={p.y} r={5} fill="#EF4444" stroke="rgba(255,255,255,0.85)" strokeWidth="1.5" />
          <circle cx={p.x} cy={p.y} r={2} fill="white" />
          <text x={p.x+7} y={p.y+4} fontSize="10" fill="#FCA5A5"
            fontFamily="'DM Mono', monospace" fontWeight="600">{key}</text>
        </g>
      ))}
      <rect x="6" y="292" width="134" height="62" rx="5" fill="rgba(0,0,0,0.62)" />
      {[
        { color: '#10B981', dash: '5,3', label: 'SN / NA Plane' },
        { color: '#3B82F6', dash: '5,3', label: 'NB Line' },
        { color: '#F59E0B', dash: '',    label: 'AB Line (ANB)' },
        { color: '#A855F7', dash: '5,3', label: 'Mandibular Plane' },
      ].map((item, i) => (
        <g key={item.label} transform={`translate(12, ${300 + i * 14})`}>
          <line x1="0" y1="5" x2="18" y2="5" stroke={item.color} strokeWidth="2"
            strokeDasharray={item.dash || undefined} />
          <text x="22" y="9" fontSize="8.5" fill="#D1D5DB" fontFamily="sans-serif">{item.label}</text>
        </g>
      ))}
      <text x="150" y="354" fontSize="9" fill="#6B7280" fontFamily="sans-serif" textAnchor="middle">
        Lateral Cephalometric Radiograph (Mô phỏng)
      </text>
    </svg>
  )
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════════════════════════
interface DiagnosisDashboardProps {
  onBack?: () => void
  uploadedImages?: Record<string, { file: File; preview: string } | null>
}

export default function DiagnosisDashboard({ onBack, uploadedImages }: DiagnosisDashboardProps) {
  const [data, setData]             = useState<FullReportResponse | null>(null)
  const [loading, setLoading]       = useState(false)
  const [error, setError]           = useState<string | null>(null)
  const [zoom, setZoom]             = useState(1)
  const [activeMode, setActiveMode] = useState<string | null>(null)
  const [scanDateTime, setScanDateTime] = useState<Date | null>(null)
  const [panoramicNatural, setPanoramicNatural] = useState({ w: 1, h: 1 })
  const [cephNatural, setCephNatural]           = useState({ w: 1, h: 1 })
  const [lrcImageUrl, setLrcImageUrl]           = useState<string | null>(null)
  const [lrcData, setLrcData]                   = useState<LrcAnalysis | null>(null)
  const [cephAiAnalysis, setCephAiAnalysis]     = useState<CephAiAnalysis | null>(null)

  const panoramicPreview = uploadedImages?.['panorama']?.preview ?? null
  const cephPreview      = uploadedImages?.['lateral-skull']?.preview ?? null
  const hasPanoramic     = Boolean(panoramicPreview)
  const hasCeph          = Boolean(cephPreview)

  const runAnalysis = useCallback(async () => {
    const panoramicFile = uploadedImages?.['panorama']?.file
    if (!panoramicFile) return
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('panoramic_file', panoramicFile)
      const cephFile = uploadedImages?.['lateral-skull']?.file
      if (cephFile) form.append('ceph_file', cephFile)
      const res = await axios.post<FullReportResponse>(`${API_URL}?fdi_conf=0.4&cv_conf=0.3`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setData(res.data)
      setScanDateTime(new Date())
      // Call LRC endpoints for ceph analysis
      const cephFileForLrc = uploadedImages?.['lateral-skull']?.file
      if (cephFileForLrc) {
        try {
          const lrcImgForm  = new FormData()
          lrcImgForm.append('file', cephFileForLrc)
          const lrcDataForm = new FormData()
          lrcDataForm.append('file', cephFileForLrc)
          const lrcAiForm   = new FormData()
          lrcAiForm.append('file', cephFileForLrc)
          const [imgRes, dataRes, aiRes] = await Promise.all([
            axios.post(LRC_IMAGE_API, lrcImgForm, { responseType: 'blob', headers: { 'Content-Type': 'multipart/form-data' } }),
            axios.post<LrcAnalysis>(LRC_DATA_API, lrcDataForm, { headers: { 'Content-Type': 'multipart/form-data' } }),
            axios.post<CephAiAnalysis>(CEPH_AI_API, lrcAiForm, { headers: { 'Content-Type': 'multipart/form-data' } }),
          ])
          const blob = new Blob([imgRes.data as BlobPart], { type: 'image/png' })
          setLrcImageUrl(URL.createObjectURL(blob))
          setLrcData(dataRes.data)
          setCephAiAnalysis(aiRes.data)
        } catch {
          // LRC analysis failed silently; ceph section falls back gracefully
        }
      }
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const detail = err.response?.data?.detail
        setError(typeof detail === 'string' ? detail : `Lỗi kết nối: ${err.message}`)
      } else {
        setError('Đã xảy ra lỗi không xác định.')
      }
    } finally {
      setLoading(false)
    }
  }, [uploadedImages])

  useEffect(() => {
    if (hasPanoramic) runAnalysis()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (!hasPanoramic && !data) return <EmptyState onBack={onBack} />

  const summary           = data?.summary
  const panoramicAnalysis = data?.panoramic_analysis ?? []
  const cephAnalysis      = data?.ceph_analysis ?? null
  const consultation      = data?.consultation ?? null
  const teethDetails      = data?.teeth_details ?? []
  const detectBtns        = ['Detect PHR', 'Detect PV', 'Detect FDI']

  const lrcMetricRows = lrcData
    ? (Object.keys(CEPH_NORMS) as (keyof typeof CEPH_NORMS)[]).map(key => ({
        key,
        display: CEPH_NORMS[key].display,
        value:   lrcData.angles[key],
        status:  classifyMetric(key, lrcData.angles[key]),
      }))
    : []

  const lrcStatus: MStatus    = lrcData ? classifyMetric('ANB', lrcData.angles.ANB) : 'normal'
  const lrcConclusion         = lrcData ? getLrcConclusion(lrcData.diagnosis, lrcData.angles.ANB) : null

  const scanDateStr = scanDateTime
    ? scanDateTime.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' })
    : '--'
  const scanTimeStr = scanDateTime
    ? scanDateTime.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true })
    : '--:--'

  return (
    <div className="min-h-screen bg-[#F3F4F6] pb-16" style={{ fontFamily: "'Sora', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');
        .thin-scroll::-webkit-scrollbar{width:4px}
        .thin-scroll::-webkit-scrollbar-track{background:transparent}
        .thin-scroll::-webkit-scrollbar-thumb{background:#D1D5DB;border-radius:99px}
      `}</style>

      {loading && <LoadingOverlay />}

      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-[1360px] mx-auto px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            {onBack && (
              <>
                <button
                  onClick={onBack}
                  className="flex items-center gap-1.5 text-[12.5px] font-semibold text-teal-700 hover:text-teal-500 transition-colors"
                >
                  <BackIcon />Quay lại
                </button>
                <div className="w-px h-7 bg-gray-200" />
              </>
            )}
            <div
              className="w-10 h-10 rounded-full flex items-center justify-center text-white font-bold text-[14px] shrink-0"
              style={{ background: 'linear-gradient(135deg, #0D9488, #14B8A6)' }}
            >NA</div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-[15px] font-bold text-gray-800">Nguyễn Văn An</h2>
                {data && (
                  <span className="text-[10px] font-bold bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full">
                    ĐÃ PHÂN TÍCH
                  </span>
                )}
              </div>
              <p className="text-[11.5px] text-gray-400">Phân tích nha khoa AI — Toàn diện</p>
            </div>
          </div>
          <div className="flex items-center gap-6">
            <div className="text-right">
              <p className="text-[11px] text-gray-400">Ngày phân tích</p>
              <p className="text-[13px] font-bold text-gray-700">{scanDateStr}</p>
            </div>
            <div className="w-px h-8 bg-gray-200" />
            <div className="text-right">
              <p className="text-[11px] text-gray-400">Thời gian</p>
              <p className="text-[13px] font-bold text-gray-700" style={{ fontFamily: "'DM Mono', monospace" }}>
                {scanTimeStr}
              </p>
            </div>
          </div>
        </div>
      </div>

      <main className="max-w-[1360px] mx-auto px-8 py-6 flex flex-col gap-6">

        {error && (
          <div className="flex items-center gap-3 bg-red-50 border border-red-200 rounded-xl px-5 py-3">
            <WarningIcon size={18} color="#EF4444" />
            <p className="text-[13px] text-red-700 flex-1 font-medium">{error}</p>
            <button
              onClick={() => setError(null)}
              className="text-red-400 hover:text-red-600 font-bold text-lg leading-none"
            >×</button>
          </div>
        )}


        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <SummaryCard
            label="Tổng số răng"
            value={summary?.total_teeth ?? 0}
            sub="Chụp toàn hàm"
            icon={<ToothIcon size={19} color="#9CA3AF" />}
          />
          <SummaryCard
            label="Răng khỏe mạnh"
            value={summary?.healthy ?? 0}
            sub="Không cần can thiệp"
            icon={<HeartIcon size={17} color="#10B981" />}
            valueColor="#10B981" bg="#F0FDF4" border="#6EE7B7"
          />
          <SummaryCard
            label="Răng sâu"
            value={summary?.cavities ?? 0}
            sub="Giai đoạn sớm"
            icon={<WarningIcon size={17} color="#F59E0B" />}
            valueColor="#F59E0B" bg="#FFFBEB" border="#FDE68A"
          />
          <SummaryCard
            label="Răng có vấn đề"
            value={summary?.needs_treatment ?? 0}
            sub="Ưu tiên cao"
            icon={<WarningIcon size={17} color="#EF4444" />}
            valueColor="#EF4444" bg="#FEF2F2" border="#FECACA"
          />
        </div>

        <div className="flex gap-6 items-stretch">
          <div className="w-[65%] bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden flex flex-col">
            <BlockHeader
              icon={<ToothIcon size={15} color="white" />}
              title="Panoramic X-Ray Analysis"
              right={
                panoramicAnalysis.length > 0
                  ? <span className="text-[11px] font-semibold bg-white/20 text-white px-3 py-0.5 rounded-full">
                      {panoramicAnalysis.length} răng bất thường
                    </span>
                  : undefined
              }
            />
            <div className="flex-1 relative bg-black overflow-hidden min-h-0 flex items-center justify-center">
              <div
                className="w-full origin-center transition-transform duration-300"
                style={{ transform: `scale(${zoom})` }}
              >
                {panoramicPreview ? (
                  <div className="relative w-full">
                    <img
                      src={panoramicPreview}
                      alt="Panoramic X-Ray"
                      className="w-full h-auto block"
                      onLoad={e => {
                        const img = e.currentTarget
                        setPanoramicNatural({ w: img.naturalWidth, h: img.naturalHeight })
                      }}
                    />
                    {panoramicNatural.w > 1 && panoramicAnalysis.map(tooth => (
                      <ToothBoundingBox
                        key={tooth.tooth_number}
                        tooth={tooth}
                        naturalW={panoramicNatural.w}
                        naturalH={panoramicNatural.h}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="relative w-full flex items-center justify-center" style={{ height: 320 }}>
                    <svg viewBox="0 0 820 260" width="96%" style={{ opacity: 0.44 }}>
                      <path d="M68 158 Q218 72 410 67 Q602 72 752 158 Q728 194 658 200 L578 200 Q550 218 538 238 L502 238 Q488 218 464 214 L356 214 Q332 218 318 238 L282 238 Q268 218 242 200 L162 200 Q92 194 68 158Z"
                        fill="none" stroke="#4B5563" strokeWidth="2.5" />
                    </svg>
                    <p className="absolute bottom-3 text-[11px] text-gray-500 tracking-wide">Phim Panorama (Mô phỏng)</p>
                  </div>
                )}
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 px-4 py-2.5 bg-gray-50 border-t border-gray-100 shrink-0">
              {detectBtns.map(btn => (
                <button
                  key={btn}
                  onClick={() => setActiveMode(p => p === btn ? null : btn)}
                  className="text-[12px] font-semibold px-3 py-1.5 rounded-lg transition-all"
                  style={{
                    color: activeMode === btn ? '#0D9488' : '#6B7280',
                    background: activeMode === btn ? '#CCFBF1' : 'transparent',
                  }}
                >{btn}</button>
              ))}
              <div className="ml-auto flex items-center gap-1.5">
                <button
                  onClick={() => setZoom(z => Math.min(z + 0.2, 2.8))}
                  className="flex items-center gap-1.5 text-[12px] font-semibold text-gray-500 hover:text-teal-600 px-2.5 py-1.5 rounded-lg hover:bg-teal-50 transition-colors"
                ><ZoomInIcon />Zoom in</button>
                <button
                  onClick={() => setZoom(z => Math.max(z - 0.2, 0.4))}
                  className="flex items-center gap-1.5 text-[12px] font-semibold text-gray-500 hover:text-teal-600 px-2.5 py-1.5 rounded-lg hover:bg-teal-50 transition-colors"
                ><ZoomOutIcon />Zoom out</button>
                <span
                  className="text-[11px] text-gray-400 min-w-[38px] text-right"
                  style={{ fontFamily: "'DM Mono', monospace" }}
                >{Math.round(zoom * 100)}%</span>
              </div>
            </div>
          </div>

          <div className="w-[35%] flex flex-col gap-4 self-stretch">
            {[
              { id: 'upper-jaw', label: 'Ảnh mặt nhai hàm trên' },
              { id: 'lower-jaw', label: 'Ảnh mặt nhai hàm dưới' },
            ].map(slot => (
              <ClinicalCard key={slot.id} label={slot.label} preview={uploadedImages?.[slot.id]?.preview} />
            ))}
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
          <BlockHeader
            icon={<BrainIcon />}
            title="Tư vấn Sức khỏe Răng miệng"
            right={<span className="text-[11px] font-semibold bg-white/20 text-white px-3 py-0.5 rounded-full">AI-Powered</span>}
          />
          {consultation ? (
            <div className="flex flex-wrap gap-6 p-6">
              <div className="flex-1 min-w-[260px]">
                <p className="text-[10.5px] font-bold text-teal-700 uppercase tracking-widest mb-3">Đánh giá tổng thể</p>
                <ul className="flex flex-col gap-2.5">
                  {consultation.overall_assessment.map((item, i) => (
                    <li key={i} className="flex gap-2.5 text-[12.5px] text-gray-700 leading-relaxed">
                      <span className="w-1.5 h-1.5 rounded-full bg-teal-400 mt-[7px] shrink-0" />{item}
                    </li>
                  ))}
                </ul>
              </div>
              <div className="w-px bg-gray-100 self-stretch hidden md:block" />
              <div className="flex-1 min-w-[260px]">
                <p className="text-[10.5px] font-bold text-red-600 uppercase tracking-widest mb-3">Vấn đề chính</p>
                <ul className="flex flex-col gap-4">
                  {consultation.main_issues.map((item, i) => (
                    <li key={i} className="flex flex-col gap-1">
                      <div className="flex gap-2.5 items-start">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-400 mt-[7px] shrink-0" />
                        <p className="text-[12.5px] font-semibold text-gray-800">{item.issue}</p>
                      </div>
                      <p className="text-[11.5px] text-gray-500 pl-4 leading-relaxed">{item.detail}</p>
                      <div className="ml-4 bg-red-50 border border-red-100 rounded-lg px-3 py-1.5">
                        <p className="text-[11.5px] font-semibold text-red-700">{item.recommendation}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 py-12">
              <div className="w-14 h-14 rounded-full flex items-center justify-center" style={{ background: '#F1F5F9' }}>
                <BrainIcon size={22} color="#94A3B8" />
              </div>
              <p className="text-[13px] font-semibold text-gray-400">Tư vấn AI sẽ xuất hiện sau khi phân tích</p>
            </div>
          )}
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
          <BlockHeader
            icon={<WarningIcon size={16} color="white" />}
            title="Chi tiết Phân tích Từng Răng"
            right={
              teethDetails.length > 0
                ? <span className="text-[11px] font-bold bg-white/20 text-white px-3 py-0.5 rounded-full">
                    {teethDetails.filter(t => t.status === 'NEEDS_TREATMENT').length} cần điều trị
                    {teethDetails.filter(t => t.status === 'ALREADY_TREATED').length > 0 &&
                      ` · ${teethDetails.filter(t => t.status === 'ALREADY_TREATED').length} đã điều trị`}
                  </span>
                : undefined
            }
          />
          <div className="p-5">
            {teethDetails.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {teethDetails.map((tooth, i) => <ToothCard key={`${tooth.tooth_number}-${i}`} tooth={tooth} />)}
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3 py-10">
                {data ? (
                  <>
                    <div className="w-14 h-14 rounded-full flex items-center justify-center" style={{ background: '#F0FDF4' }}>
                      <HeartIcon size={24} color="#10B981" />
                    </div>
                    <p className="text-[13px] font-semibold text-emerald-600">Không phát hiện răng cần điều trị đặc biệt!</p>
                  </>
                ) : (
                  <>
                    <div className="w-14 h-14 rounded-full flex items-center justify-center" style={{ background: '#F1F5F9' }}>
                      <ToothIcon size={24} color="#94A3B8" />
                    </div>
                    <p className="text-[13px] font-semibold text-gray-400">Chi tiết từng răng sẽ hiển thị sau khi phân tích</p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex gap-6 items-stretch">
          <div className="w-1/2 bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden flex flex-col">
            <BlockHeader icon={<RulerIcon />} title="Cephalometric X-Ray (Phim Sọ Nghiêng)" />
            <div className="flex-1 flex items-center justify-center relative" style={{ background: '#0F172A', minHeight: 360 }}>
              {lrcImageUrl ? (
                <img
                  src={lrcImageUrl}
                  alt="Cephalometric Analysis"
                  className="w-full h-auto block"
                />
              ) : cephPreview ? (
                <div className="relative w-full">
                  <img
                    src={cephPreview}
                    alt="Cephalometric X-Ray"
                    className="w-full h-auto block"
                    onLoad={e => {
                      const img = e.currentTarget
                      setCephNatural({ w: img.naturalWidth, h: img.naturalHeight })
                    }}
                  />
                  {cephNatural.w > 1 && cephAnalysis && (
                    <CephLines
                      landmarks={cephAnalysis.landmarks}
                      naturalW={cephNatural.w}
                      naturalH={cephNatural.h}
                    />
                  )}
                  {cephNatural.w > 1 && cephAnalysis?.landmarks
                    .map(point => (
                    <LandmarkDot
                      key={point.name}
                      point={point}
                      naturalW={cephNatural.w}
                      naturalH={cephNatural.h}
                    />
                  ))}
                  {cephAnalysis && (
                    <div
                      className="absolute pointer-events-none select-none"
                      style={{ top: 10, left: 10, zIndex: 20 }}
                    >
                      <p
                        className="text-[13px] font-bold mb-1 text-center"
                        style={{ color: '#00E5FF', textShadow: '0 1px 4px rgba(0,0,0,0.9)', fontFamily: "'DM Mono', monospace" }}
                      >
                        Chẩn Đoán Hô/Móm (ANB = {cephAnalysis.metrics.ANB.toFixed(2)}°)
                      </p>
                      {(['SNA', 'SNB', 'ANB'] as const).map(key => (
                        <p
                          key={key}
                          className="text-[12px] font-semibold leading-snug"
                          style={{ color: '#00E5FF', textShadow: '0 1px 4px rgba(0,0,0,0.9)', fontFamily: "'DM Mono', monospace" }}
                        >
                          {key}: {cephAnalysis.metrics[key].toFixed(2)} deg
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <CephalometricPlaceholder />
              )}
            </div>
            {/* Landmark legend */}
            {!lrcImageUrl && cephAnalysis && cephAnalysis.landmarks.length > 0 && (
              <div className="border-t border-gray-100 bg-gray-50 px-4 py-3">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">Giải thích điểm mốc (hover lên ảnh để xem chi tiết)</p>
                <div className="flex flex-wrap gap-x-4 gap-y-1">
                  {cephAnalysis.landmarks.map(pt => {
                    const info = LM_INFO[pt.name]
                    const color = LANDMARK_COLORS[pt.name] ?? '#EF4444'
                    return (
                      <div key={pt.name} className="flex items-center gap-1 group/leg relative">
                        <div className="w-2 h-2 rounded-full border border-white shrink-0" style={{ background: color }} />
                        <span className="text-[10px] font-bold text-gray-600" style={{ fontFamily: "'DM Mono', monospace" }}>
                          {pt.name}
                        </span>
                        {info && (
                          <span className="text-[10px] text-gray-400">— {info.full}</span>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>

          <div className="w-1/2 bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden flex flex-col">
            <BlockHeader
              icon={<RulerIcon />}
              title="Kết Quả Đo Đạc Cephalometric"
              right={<span className="text-[11px] font-semibold bg-white/20 text-white px-3 py-0.5 rounded-full">Chẩn đoán Hô / Móm</span>}
            />
            <div className="flex-1 p-5 flex flex-col gap-4 overflow-auto thin-scroll">
              {lrcData ? (
                <>
                  <div
                    className="rounded-xl px-4 py-3 border"
                    style={{ background: mStatusBg(lrcStatus), borderColor: lrcStatus === 'normal' ? '#6EE7B7' : '#FECACA' }}
                  >
                    <p className="text-[10.5px] font-bold uppercase tracking-widest mb-1" style={{ color: mStatusColor(lrcStatus) }}>
                      Kết luận Chẩn đoán
                    </p>
                    <p className="text-[14px] font-bold text-gray-800">{lrcConclusion!.conclusion}</p>
                    {lrcConclusion!.conclusion_detail && (
                      <p className="text-[12px] text-gray-500 mt-0.5">{lrcConclusion!.conclusion_detail}</p>
                    )}
                  </div>
                  <div className="flex-1 overflow-auto thin-scroll">
                    <table className="w-full text-[12.5px] border-collapse">
                      <thead>
                        <tr className="border-b-2 border-gray-100">
                          {['Chỉ số', 'Giá trị', 'Chuẩn', 'Kết quả'].map((h, i) => (
                            <th key={i} className={`pb-2.5 text-[11px] font-bold text-gray-400 uppercase tracking-wide ${i === 0 ? 'text-left' : 'text-center'}`}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {lrcMetricRows.map((m, i) => (
                          <tr key={m.key} className={`border-b border-gray-50 ${i % 2 === 0 ? 'bg-gray-50/60' : ''}`}>
                            <td className="py-2.5 pr-3">
                              <span className="font-bold text-gray-700" style={{ fontFamily: "'DM Mono', monospace" }}>{m.display}</span>
                            </td>
                            <td className="py-2.5 text-center">
                              <span className="font-bold" style={{ fontFamily: "'DM Mono', monospace", color: mStatusColor(m.status) }}>
                                {m.value.toFixed(2)}°
                              </span>
                            </td>
                            <td className="py-2.5 text-center text-gray-400 text-[12px]">{CEPH_NORMS[m.key].label}</td>
                            <td className="py-2.5 text-center">
                              <span
                                className="text-[10px] font-bold px-2.5 py-0.5 rounded-full"
                                style={{ background: mStatusBg(m.status), color: mStatusColor(m.status) }}
                              >
                                {mStatusLabel(m.status)}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center gap-3 py-12">
                  <div className="w-14 h-14 rounded-full flex items-center justify-center" style={{ background: '#F1F5F9' }}>
                    <RulerIcon size={22} color="#94A3B8" />
                  </div>
                  <p className="text-[13px] font-semibold text-gray-400 text-center">
                    {hasCeph ? 'Nhấn "Bắt đầu Phân tích AI" để đo các chỉ số' : 'Chưa tải ảnh Cephalometric'}
                  </p>
                  <p className="text-[11.5px] text-gray-300 text-center max-w-[220px]">
                    Ảnh sọ nghiêng là tùy chọn nhưng giúp chẩn đoán hô/móm chính xác hơn.
                  </p>
                </div>
              )}
              <div className="rounded-xl bg-gray-50 px-4 py-3 flex flex-wrap gap-x-5 gap-y-2">
                {[
                  { color: '#22C55E', label: 'SN / NA / NB / AB' },
                  { color: '#F59E0B', label: 'Frankfort Horizontal' },
                  { color: '#3B82F6', label: 'Palatal Plane' },
                  { color: '#A855F7', label: 'Mandibular Plane' },
                  { color: '#F97316', label: 'Incisor Axes' },
                  { color: '#14B8A6', label: 'Facial Line (N-Pog)' },
                ].map(item => (
                  <div key={item.label} className="flex items-center gap-1.5">
                    <div className="w-5 h-0.5 rounded-full" style={{ background: item.color }} />
                    <span className="text-[10.5px] text-gray-500">{item.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {cephAiAnalysis && <CephAiCard analysis={cephAiAnalysis} />}

      </main>
    </div>
  )
}
