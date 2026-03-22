import { useEffect, useMemo, useState } from 'react'
import ProbabilityChart from '../components/charts/ProbabilityChart'
import { simulateSeason } from '../services/api'

function SeasonSimulation() {
  const [iterations, setIterations] = useState(2000)
  const [volatility, setVolatility] = useState(1.0)
  const [cutoffDate, setCutoffDate] = useState('')
  const [result, setResult] = useState(null)
  const [chartMetric, setChartMetric] = useState('titleProb')
  const [isLoading, setIsLoading] = useState(true)
  const [isRunning, setIsRunning] = useState(false)
  const [error, setError] = useState('')

  async function runSimulation(options = {}) {
    const nextIterations = options.iterations ?? iterations
    const nextVolatility = options.volatility ?? volatility
    const nextCutoffDate = options.cutoffDate ?? cutoffDate

    setError('')
    setIsRunning(true)
    try {
      const payload = await simulateSeason({
        iterations: nextIterations,
        volatility: nextVolatility,
        cutoffDate: nextCutoffDate,
      })
      setResult(payload)
    } catch (simulationError) {
      setError(simulationError.message || 'Simulation failed.')
    } finally {
      setIsLoading(false)
      setIsRunning(false)
    }
  }

  useEffect(() => {
    runSimulation({ iterations, volatility, cutoffDate })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const topTeams = useMemo(() => (result?.teams || []).slice(0, 8), [result])
  const chartTeams = useMemo(() => {
    const teams = [...(result?.teams || [])]
    return teams
      .sort((a, b) => Number(b[chartMetric] || 0) - Number(a[chartMetric] || 0))
      .slice(0, 10)
  }, [chartMetric, result])

  const champion = useMemo(() => {
    if (!result?.teams?.length) return null
    return [...result.teams].sort((a, b) => b.titleProb - a.titleProb)[0]
  }, [result])

  const top4Lock = useMemo(() => {
    if (!result?.teams?.length) return null
    const championTeam = champion?.team
    const filtered = result.teams.filter((team) => team.team !== championTeam)
    if (filtered.length === 0) return result.teams[0]
    return [...filtered].sort((a, b) => b.top4Prob - a.top4Prob)[0]
  }, [champion?.team, result])

  const relegationRisk = useMemo(() => {
    if (!result?.teams?.length) return null
    return [...result.teams].sort((a, b) => b.relegationProb - a.relegationProb)[0]
  }, [result])

  const metricConfig = useMemo(() => {
    const map = {
      titleProb: {
        label: 'Title Probability',
        colorClass: 'bg-emerald-400',
        trackClass: 'bg-emerald-900/30',
      },
      top4Prob: {
        label: 'Top 4 Probability',
        colorClass: 'bg-cyan-400',
        trackClass: 'bg-cyan-900/30',
      },
      relegationProb: {
        label: 'Relegation Probability',
        colorClass: 'bg-rose-400',
        trackClass: 'bg-rose-900/30',
      },
    }
    return map[chartMetric]
  }, [chartMetric])

  return (
    <section className="space-y-5">
      <article className="panel-glass rounded-2xl border border-white/10 p-6">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h3 className="text-2xl font-semibold text-white">Season Simulation</h3>
            <p className="mt-1 text-sm text-slate-300">
              Monte Carlo simulation of final table outcomes based on current season performance.
            </p>
          </div>

          <form
            className="flex flex-wrap items-end gap-3"
            onSubmit={(event) => {
              event.preventDefault()
              runSimulation({ iterations, volatility, cutoffDate })
            }}
          >
            <label className="block">
              <span className="mb-1 block text-xs uppercase tracking-[0.12em] text-slate-400">Iterations</span>
              <input
                type="number"
                min={100}
                max={10000}
                step={100}
                value={iterations}
                onChange={(event) => setIterations(Number(event.target.value || 1000))}
                className="w-32 rounded-xl border border-white/10 bg-slate-900/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400/40"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs uppercase tracking-[0.12em] text-slate-400">Volatility</span>
              <input
                type="number"
                min={0.6}
                max={1.8}
                step={0.1}
                value={volatility}
                onChange={(event) => setVolatility(Number(event.target.value || 1.0))}
                className="w-28 rounded-xl border border-white/10 bg-slate-900/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400/40"
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs uppercase tracking-[0.12em] text-slate-400">As Of Date</span>
              <input
                type="date"
                value={cutoffDate}
                onChange={(event) => setCutoffDate(event.target.value)}
                className="w-40 rounded-xl border border-white/10 bg-slate-900/60 px-3 py-2 text-sm text-slate-100 outline-none focus:border-cyan-400/40"
              />
            </label>
            <button
              type="submit"
              disabled={isRunning}
              className="rounded-xl border border-cyan-400/40 bg-cyan-400/10 px-4 py-2.5 text-sm font-semibold text-cyan-100 transition-colors hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isRunning ? 'Simulating...' : 'Run Simulation'}
            </button>
          </form>
        </div>
      </article>

      {error && (
        <p className="rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">
          {error}
        </p>
      )}

      {isLoading ? (
        <section className="grid gap-4 md:grid-cols-3">
          <div className="panel-glass h-28 animate-pulse rounded-2xl border border-white/10" />
          <div className="panel-glass h-28 animate-pulse rounded-2xl border border-white/10" />
          <div className="panel-glass h-28 animate-pulse rounded-2xl border border-white/10" />
        </section>
      ) : !result?.teams?.length ? (
        <article className="panel-glass rounded-2xl border border-dashed border-white/20 p-6 text-sm text-slate-400">
          No simulation output is available. Try running the simulation again.
        </article>
      ) : (
        <>
          <section className="grid gap-4 md:grid-cols-3">
            <article className="panel-glass rounded-2xl border border-white/10 p-5">
              <p className="text-sm text-slate-400">Most Likely Champion</p>
              <h4 className="mt-2 text-2xl font-semibold text-white">{champion?.team || '-'}</h4>
              <p className="mt-1 text-sm text-emerald-300">Title probability {champion?.titleProb || 0}%</p>
            </article>

            <article className="panel-glass rounded-2xl border border-white/10 p-5">
              <p className="text-sm text-slate-400">Top 4 Safest Team</p>
              <h4 className="mt-2 text-2xl font-semibold text-white">{top4Lock?.team || '-'}</h4>
              <p className="mt-1 text-sm text-cyan-300">Top 4 probability {top4Lock?.top4Prob || 0}%</p>
            </article>

            <article className="panel-glass rounded-2xl border border-white/10 p-5">
              <p className="text-sm text-slate-400">Highest Relegation Risk</p>
              <h4 className="mt-2 text-2xl font-semibold text-white">{relegationRisk?.team || '-'}</h4>
              <p className="mt-1 text-sm text-rose-300">
                Relegation probability {relegationRisk?.relegationProb || 0}%
              </p>
            </article>
          </section>

          <article className="panel-glass rounded-2xl border border-white/10 p-5">
            <div className="mb-4 flex items-center justify-between">
              <h4 className="text-lg font-semibold text-white">Projected Table Snapshot</h4>
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-full border border-indigo-400/30 bg-indigo-400/10 px-3 py-1 text-[10px] uppercase tracking-[0.12em] text-indigo-200">
                  {result?.iterations || iterations} runs
                </span>
                <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-[10px] uppercase tracking-[0.12em] text-cyan-200">
                  volatility {Number(result?.volatility || volatility).toFixed(1)}
                </span>
                {result?.cutoffDate && (
                  <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-3 py-1 text-[10px] uppercase tracking-[0.12em] text-amber-200">
                    as of {result.cutoffDate}
                  </span>
                )}
              </div>
            </div>

            <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/20">
              <table className="w-full border-collapse text-sm">
                <thead>
                  <tr className="bg-white/5 text-left text-xs uppercase tracking-[0.14em] text-slate-400">
                    <th className="px-3 py-3">Team</th>
                    <th className="px-3 py-3 text-center">Avg Pts</th>
                    <th className="px-3 py-3 text-center">Likely Pos</th>
                    <th className="px-3 py-3 text-center">Title %</th>
                    <th className="px-3 py-3 text-center">Top 4 %</th>
                    <th className="px-3 py-3 text-center">Rel %</th>
                  </tr>
                </thead>
                <tbody>
                  {topTeams.map((team) => (
                    <tr key={team.team} className="border-b border-white/5 text-slate-200 hover:bg-white/5">
                      <td className="px-3 py-3 font-medium text-slate-100">{team.team}</td>
                      <td className="px-3 py-3 text-center">{team.avgPoints}</td>
                      <td className="px-3 py-3 text-center">{team.mostLikelyPosition}</td>
                      <td className="px-3 py-3 text-center text-emerald-300">{team.titleProb}%</td>
                      <td className="px-3 py-3 text-center text-cyan-300">{team.top4Prob}%</td>
                      <td className="px-3 py-3 text-center text-rose-300">{team.relegationProb}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>

          <article className="panel-glass rounded-2xl border border-white/10 p-5">
            <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <h4 className="text-lg font-semibold text-white">Probability Chart View</h4>
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setChartMetric('titleProb')}
                  className={[
                    'rounded-lg px-3 py-1.5 text-xs font-medium transition-colors',
                    chartMetric === 'titleProb'
                      ? 'bg-emerald-500/20 text-emerald-200 ring-1 ring-emerald-400/40'
                      : 'bg-white/5 text-slate-300 hover:bg-white/10',
                  ].join(' ')}
                >
                  Title
                </button>
                <button
                  type="button"
                  onClick={() => setChartMetric('top4Prob')}
                  className={[
                    'rounded-lg px-3 py-1.5 text-xs font-medium transition-colors',
                    chartMetric === 'top4Prob'
                      ? 'bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-400/40'
                      : 'bg-white/5 text-slate-300 hover:bg-white/10',
                  ].join(' ')}
                >
                  Top 4
                </button>
                <button
                  type="button"
                  onClick={() => setChartMetric('relegationProb')}
                  className={[
                    'rounded-lg px-3 py-1.5 text-xs font-medium transition-colors',
                    chartMetric === 'relegationProb'
                      ? 'bg-rose-500/20 text-rose-200 ring-1 ring-rose-400/40'
                      : 'bg-white/5 text-slate-300 hover:bg-white/10',
                  ].join(' ')}
                >
                  Relegation
                </button>
              </div>
            </div>

            <p className="mb-4 text-xs uppercase tracking-[0.12em] text-slate-400">
              Top 10 by {metricConfig.label}
            </p>

            <ProbabilityChart
              teams={chartTeams}
              metric={chartMetric}
              colorClass={metricConfig.colorClass}
              trackClass={metricConfig.trackClass}
              valueFormatter={(value) => `${value.toFixed(1)}%`}
            />
          </article>
        </>
      )}
    </section>
  )
}

export default SeasonSimulation
