import { useEffect, useMemo, useState } from 'react'
import { getTeamAvailability, updateTeamAvailability } from '../services/api'

function TeamAvailability() {
  const [rows, setRows] = useState([])
  const [savedRows, setSavedRows] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  useEffect(() => {
    let active = true

    async function loadRows() {
      setIsLoading(true)
      setError('')
      try {
        const payload = await getTeamAvailability()
        if (!active) return
        setRows(payload)
        setSavedRows(payload)
      } catch (loadError) {
        if (!active) return
        setError(loadError.message || 'Could not load team availability.')
      } finally {
        if (active) setIsLoading(false)
      }
    }

    loadRows()
    return () => {
      active = false
    }
  }, [])

  const hasChanges = useMemo(
    () => JSON.stringify(rows) !== JSON.stringify(savedRows),
    [rows, savedRows],
  )

  function updateCell(index, key, value) {
    setSuccess('')
    setError('')
    setRows((previous) => {
      const next = [...previous]
      next[index] = { ...next[index], [key]: value }
      return next
    })
  }

  async function handleSave() {
    setIsSaving(true)
    setError('')
    setSuccess('')
    try {
      const payload = rows.map((row) => ({
        team: String(row.team || '').trim(),
        injured: Number(row.injured || 0),
        suspended: Number(row.suspended || 0),
        key_impact: Math.max(0, Math.min(1, Number(row.key_impact || 0))),
      }))
      await updateTeamAvailability(payload)
      setSavedRows(payload)
      setRows(payload)
      setSuccess('Saved. Predictions now use the updated availability values.')
    } catch (saveError) {
      setError(saveError.message || 'Save failed.')
    } finally {
      setIsSaving(false)
    }
  }

  function handleReset() {
    setRows(savedRows)
    setError('')
    setSuccess('')
  }

  if (isLoading) {
    return (
      <section className="space-y-4">
        <div className="panel-glass h-28 animate-pulse rounded-2xl border border-white/10" />
        <div className="panel-glass h-64 animate-pulse rounded-2xl border border-white/10" />
      </section>
    )
  }

  return (
    <section className="space-y-5">
      <article className="panel-glass rounded-2xl border border-white/10 p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-2xl font-semibold text-white">Team Availability</h3>
            <p className="mt-1 text-sm text-slate-300">
              Update injuries, suspensions, and key-player impact. Changes apply to live predictions.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleReset}
              disabled={!hasChanges || isSaving}
              className="rounded-xl border border-white/20 bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Reset
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={!hasChanges || isSaving}
              className="rounded-xl border border-cyan-400/40 bg-cyan-400/10 px-4 py-2 text-sm font-semibold text-cyan-100 transition-colors hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isSaving ? 'Saving...' : 'Save Changes'}
            </button>
          </div>
        </div>
      </article>

      {error && (
        <p className="rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">
          {error}
        </p>
      )}
      {success && (
        <p className="rounded-lg border border-emerald-400/30 bg-emerald-400/10 px-3 py-2 text-sm text-emerald-200">
          {success}
        </p>
      )}

      <article className="panel-glass overflow-hidden rounded-2xl border border-white/10">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="bg-white/5 text-left text-xs uppercase tracking-[0.12em] text-slate-400">
                <th className="px-4 py-3">Team</th>
                <th className="px-4 py-3 text-center">Injured</th>
                <th className="px-4 py-3 text-center">Suspended</th>
                <th className="px-4 py-3 text-center">Key Impact</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={row.team} className="border-b border-white/5 text-slate-200 hover:bg-white/[0.03]">
                  <td className="px-4 py-3 font-medium text-slate-100">{row.team}</td>
                  <td className="px-4 py-3 text-center">
                    <input
                      type="number"
                      min={0}
                      step={1}
                      value={row.injured}
                      onChange={(event) => updateCell(index, 'injured', event.target.value)}
                      className="w-24 rounded-lg border border-white/15 bg-slate-900/60 px-2 py-1 text-center text-sm text-slate-100 outline-none focus:border-cyan-400/40"
                    />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <input
                      type="number"
                      min={0}
                      step={1}
                      value={row.suspended}
                      onChange={(event) => updateCell(index, 'suspended', event.target.value)}
                      className="w-24 rounded-lg border border-white/15 bg-slate-900/60 px-2 py-1 text-center text-sm text-slate-100 outline-none focus:border-cyan-400/40"
                    />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.05}
                      value={row.key_impact}
                      onChange={(event) => updateCell(index, 'key_impact', event.target.value)}
                      className="w-24 rounded-lg border border-white/15 bg-slate-900/60 px-2 py-1 text-center text-sm text-slate-100 outline-none focus:border-cyan-400/40"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  )
}

export default TeamAvailability
