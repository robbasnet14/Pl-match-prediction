import { useCallback, useEffect, useState } from 'react'
import FixturesList from '../components/fixtures/FixturesList'
import PredictionCard from '../components/prediction/PredictionCard'
import LeagueTable from '../components/table/LeagueTable'
import { mockDashboardData } from '../data/mockData'
import { getDashboardData } from '../services/api'

function Dashboard() {
  const [dashboardData, setDashboardData] = useState(mockDashboardData)
  const [source, setSource] = useState('mock')
  const [isLoading, setIsLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [hasError, setHasError] = useState(false)
  const [lastSyncedAt, setLastSyncedAt] = useState(null)

  const loadDashboard = useCallback(async (isBackgroundRefresh = false) => {
    if (isBackgroundRefresh) {
      setIsRefreshing(true)
    } else {
      setIsLoading(true)
    }

    setHasError(false)

    try {
      const result = await getDashboardData()
      setDashboardData(result.data)
      setSource(result.source)
      setLastSyncedAt(new Date().toISOString())
    } catch {
      setDashboardData(mockDashboardData)
      setSource('mock-fallback')
      setHasError(true)
      setLastSyncedAt(new Date().toISOString())
    } finally {
      if (isBackgroundRefresh) {
        setIsRefreshing(false)
      } else {
        setIsLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    loadDashboard(false)
  }, [loadDashboard])

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      loadDashboard(true)
    }, 60000)
    return () => window.clearInterval(intervalId)
  }, [loadDashboard])

  const {
    summary,
    fixtures,
    standings,
    predictions,
    dataStatus,
    liveDataStatus,
  } =
    dashboardData

  const showLiveDataWarning =
    source === 'api' &&
    (!liveDataStatus.configured ||
      !liveDataStatus.fixturesUsingExternal ||
      !liveDataStatus.standingsUsingExternal)
  const showLiveDataActive =
    source === 'api' &&
    liveDataStatus.configured &&
    liveDataStatus.fixturesUsingExternal &&
    liveDataStatus.standingsUsingExternal

  if (isLoading) {
    return (
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <div className="panel-glass h-32 animate-pulse rounded-2xl border border-white/10" />
        <div className="panel-glass h-32 animate-pulse rounded-2xl border border-white/10" />
        <div className="panel-glass h-32 animate-pulse rounded-2xl border border-white/10" />
        <div className="panel-glass h-32 animate-pulse rounded-2xl border border-white/10" />
        <div className="panel-glass h-32 animate-pulse rounded-2xl border border-white/10" />
      </section>
    )
  }

  return (
    <section className="space-y-5">
      {showLiveDataWarning && (
        <div className="rounded-xl border border-amber-300/30 bg-amber-400/10 px-4 py-3 text-sm text-amber-100">
          <p className="font-medium">Live API not fully active</p>
          <p className="mt-1 text-xs text-amber-200/90">{liveDataStatus.message}</p>
        </div>
      )}
      {showLiveDataActive && (
        <div className="rounded-xl border border-emerald-300/30 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-100">
          <p className="font-medium">External live data active</p>
          <p className="mt-1 text-xs text-emerald-200/90">
            Standings and upcoming fixtures are both coming from the external live provider.
          </p>
        </div>
      )}

      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-400">
          Last synced:{' '}
          {lastSyncedAt ? new Date(lastSyncedAt).toLocaleTimeString() : 'not yet'}
          {isRefreshing ? ' · refreshing...' : ''}
        </p>
        <button
          type="button"
          onClick={() => loadDashboard(true)}
          className="rounded-lg border border-white/20 bg-white/5 px-3 py-1.5 text-xs text-slate-200 transition-colors hover:bg-white/10"
        >
          Refresh now
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <article className="panel-glass card-hover rounded-2xl border border-white/10 p-5">
          <p className="text-sm text-slate-400">Upcoming Fixtures</p>
          <h3 className="mt-2 text-3xl font-semibold text-white">{summary.upcomingFixtures}</h3>
        </article>

        <article className="panel-glass card-hover rounded-2xl border border-white/10 p-5">
          <p className="text-sm text-slate-400">Prediction Confidence</p>
          <h3 className="mt-2 text-3xl font-semibold text-white">{summary.predictionConfidence}%</h3>
        </article>

        <article className="panel-glass card-hover rounded-2xl border border-white/10 p-5">
          <p className="text-sm text-slate-400">Live Feed</p>
          <h3
            className={[
              'mt-2 text-2xl font-semibold',
              showLiveDataActive ? 'text-emerald-300' : 'text-amber-300',
            ].join(' ')}
          >
            {showLiveDataActive ? 'Connected' : 'Limited'}
          </h3>
          <p className="mt-1 text-xs text-slate-400">
            {showLiveDataActive ? 'Fixtures and table are live.' : liveDataStatus.message}
          </p>
        </article>

        <article className="panel-glass card-hover rounded-2xl border border-white/10 p-5">
          <p className="text-sm text-slate-400">Data Status</p>
          <h3
            className={[
              'mt-2 text-2xl font-semibold',
              dataStatus.isComplete ? 'text-emerald-300' : 'text-amber-300',
            ].join(' ')}
          >
            {dataStatus.isComplete ? 'Complete' : 'Incomplete'}
          </h3>
          <p className="mt-1 text-xs text-slate-400">
            Season {dataStatus.latestSeason} · {dataStatus.latestSeasonTeams}/20 teams
          </p>
          <p className="mt-1 text-xs text-slate-500">Latest match {dataStatus.latestMatchDate}</p>
        </article>
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.35fr_1fr]">
        <article className="panel-glass rounded-2xl border border-white/10 p-5">
          <div className="mb-4 flex items-end justify-between gap-3">
            <div>
              <p className="text-sm text-slate-400">Table</p>
              <h3 className="mt-1 text-xl font-semibold text-white">Animated League Standings</h3>
            </div>
            <p className="text-xs text-slate-400">Updated {new Date(summary.updatedAt).toLocaleTimeString()}</p>
          </div>
          <LeagueTable teams={standings} />
        </article>

        <div className="grid gap-5">
          <article className="panel-glass rounded-2xl border border-white/10 p-5">
            <div className="mb-4">
              <p className="text-sm text-slate-400">Fixtures</p>
              <h3 className="mt-1 text-xl font-semibold text-white">Upcoming Matches</h3>
            </div>
            {fixtures.length > 0 ? (
              <FixturesList fixtures={fixtures} />
            ) : (
              <p className="rounded-lg border border-dashed border-white/20 bg-white/[0.02] px-3 py-2 text-sm text-slate-400">
                No fixtures available.
              </p>
            )}
          </article>

          <article className="panel-glass rounded-2xl border border-white/10 p-5">
            <p className="text-sm text-slate-400">Predictions</p>
            <h3 className="mb-4 mt-1 text-xl font-semibold text-white">Model Confidence Cards</h3>
            {hasError && (
              <p className="mb-3 rounded-lg border border-amber-300/30 bg-amber-400/10 px-3 py-2 text-xs text-amber-200">
                API unavailable. Showing fallback mock data.
              </p>
            )}
            {predictions.length > 0 ? (
              <div className="space-y-3">
                {predictions.map((prediction) => (
                  <PredictionCard key={prediction.id} prediction={prediction} />
                ))}
              </div>
            ) : (
              <p className="rounded-lg border border-dashed border-white/20 bg-white/[0.02] px-3 py-2 text-sm text-slate-400">
                No predictions available.
              </p>
            )}
          </article>
        </div>
      </div>
    </section>
  )
}

export default Dashboard
