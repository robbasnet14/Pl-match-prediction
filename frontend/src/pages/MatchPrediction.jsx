import { useEffect, useMemo, useState } from 'react'
import PredictionCard from '../components/prediction/PredictionCard'
import { getTeams, getUpcomingFixtures, predictMatch } from '../services/api'

function MatchPrediction() {
  const [teams, setTeams] = useState([])
  const [fixtures, setFixtures] = useState([])
  const [fixturesSource, setFixturesSource] = useState('unknown')
  const [homeTeam, setHomeTeam] = useState('')
  const [awayTeam, setAwayTeam] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [isLoadingTeams, setIsLoadingTeams] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    async function loadInitialData() {
      setIsLoadingTeams(true)
      setError('')

      try {
        const [allTeams, fixturesPayload] = await Promise.all([
          getTeams(),
          getUpcomingFixtures(),
        ])

        if (!active) return

        setTeams(allTeams)
        setFixtures(fixturesPayload.fixtures)
        setFixturesSource(fixturesPayload.source)

        if (allTeams.length >= 2) {
          setHomeTeam(allTeams[0])
          setAwayTeam(allTeams[1])
        }
      } catch {
        if (!active) return
        setError('Could not load teams/fixtures from API. Make sure backend is running.')
      } finally {
        if (active) setIsLoadingTeams(false)
      }
    }

    loadInitialData()
    return () => {
      active = false
    }
  }, [])

  const awayOptions = useMemo(
    () => teams.filter((team) => team !== homeTeam),
    [teams, homeTeam],
  )

  useEffect(() => {
    if (awayTeam === homeTeam && awayOptions.length > 0) {
      setAwayTeam(awayOptions[0])
    }
  }, [awayOptions, awayTeam, homeTeam])

  async function runPrediction(nextHomeTeam, nextAwayTeam) {
    setError('')

    if (!nextHomeTeam || !nextAwayTeam) {
      setError('Please select both teams.')
      return
    }

    if (nextHomeTeam === nextAwayTeam) {
      setError('Home and away teams must be different.')
      return
    }

    setIsSubmitting(true)
    try {
      const nextPrediction = await predictMatch(nextHomeTeam, nextAwayTeam)
      setPrediction(nextPrediction)
    } catch (requestError) {
      setError(requestError.message || 'Prediction request failed.')
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    await runPrediction(homeTeam, awayTeam)
  }

  async function handleQuickPick(fixture) {
    setHomeTeam(fixture.home)
    setAwayTeam(fixture.away)
    await runPrediction(fixture.home, fixture.away)
  }

  return (
    <section className="grid gap-5 lg:grid-cols-[420px_1fr]">
      <article className="panel-glass rounded-2xl border border-white/10 p-6">
        <h3 className="text-2xl font-semibold text-white">Match Prediction</h3>
        <p className="mt-2 text-sm text-slate-300">
          Select any two teams and run a live backend model prediction.
        </p>

        <div className="mt-5">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-xs uppercase tracking-[0.12em] text-slate-400">Quick Pick Upcoming</p>
            <span className="rounded-full border border-indigo-400/30 bg-indigo-400/10 px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-indigo-200">
              {fixturesSource}
            </span>
          </div>

          <div className="space-y-2">
            {isLoadingTeams && (
              <>
                <div className="h-10 animate-pulse rounded-xl border border-white/10 bg-white/5" />
                <div className="h-10 animate-pulse rounded-xl border border-white/10 bg-white/5" />
                <div className="h-10 animate-pulse rounded-xl border border-white/10 bg-white/5" />
              </>
            )}
            {!isLoadingTeams && fixtures.length === 0 && (
              <p className="rounded-xl border border-dashed border-white/20 bg-white/[0.02] px-3 py-2 text-sm text-slate-400">
                No upcoming fixtures are available right now.
              </p>
            )}
            {!isLoadingTeams &&
              fixtures.slice(0, 3).map((fixture) => (
                <button
                  key={fixture.id}
                  type="button"
                  onClick={() => handleQuickPick(fixture)}
                  disabled={isSubmitting}
                  className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-left text-sm text-slate-200 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {fixture.home} vs {fixture.away}
                </button>
              ))}
          </div>
        </div>

        <form className="mt-6 space-y-4" onSubmit={handleSubmit}>
          <label className="block">
            <span className="mb-2 block text-xs uppercase tracking-[0.12em] text-slate-400">Home Team</span>
            <select
              value={homeTeam}
              onChange={(event) => setHomeTeam(event.target.value)}
              disabled={isLoadingTeams || isSubmitting || teams.length === 0}
              className="w-full rounded-xl border border-white/10 bg-slate-900/60 px-3 py-3 text-sm text-slate-100 outline-none transition-colors focus:border-cyan-400/40"
            >
              {teams.map((team) => (
                <option key={`home-${team}`} value={team}>
                  {team}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="mb-2 block text-xs uppercase tracking-[0.12em] text-slate-400">Away Team</span>
            <select
              value={awayTeam}
              onChange={(event) => setAwayTeam(event.target.value)}
              disabled={isLoadingTeams || isSubmitting || awayOptions.length === 0}
              className="w-full rounded-xl border border-white/10 bg-slate-900/60 px-3 py-3 text-sm text-slate-100 outline-none transition-colors focus:border-cyan-400/40"
            >
              {awayOptions.map((team) => (
                <option key={`away-${team}`} value={team}>
                  {team}
                </option>
              ))}
            </select>
          </label>

          <button
            type="submit"
            disabled={isLoadingTeams || isSubmitting || !homeTeam || !awayTeam}
            className="w-full rounded-xl border border-cyan-400/40 bg-cyan-400/10 px-4 py-3 text-sm font-semibold text-cyan-100 transition-colors hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {isSubmitting ? 'Predicting...' : 'Run Prediction'}
          </button>
        </form>

        {error && (
          <p className="mt-4 rounded-lg border border-rose-400/30 bg-rose-400/10 px-3 py-2 text-sm text-rose-200">
            {error}
          </p>
        )}
      </article>

      <article className="panel-glass rounded-2xl border border-white/10 p-6">
        <h4 className="text-lg font-semibold text-white">Prediction Result</h4>
        <p className="mt-1 text-sm text-slate-400">Generated from the backend model in real time.</p>

        <div className="mt-5">
          {prediction ? (
            <PredictionCard prediction={prediction} />
          ) : (
            <div className="rounded-xl border border-dashed border-white/20 bg-white/[0.02] p-6 text-sm text-slate-400">
              Submit a match to see probabilities and likely score.
            </div>
          )}
        </div>
      </article>
    </section>
  )
}

export default MatchPrediction
