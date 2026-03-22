import { mockDashboardData } from '../data/mockData'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL

export async function getApiHealth() {
  if (!API_BASE_URL) {
    return { ok: false, reason: 'missing_api_base_url' }
  }

  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    if (!response.ok) {
      return { ok: false, reason: `status_${response.status}` }
    }
    const payload = await response.json()
    return {
      ok: Boolean(payload.ok),
      timestamp: payload.timestamp || null,
    }
  } catch {
    return { ok: false, reason: 'network_error' }
  }
}

function normalizeDashboardData(payload) {
  if (!payload || typeof payload !== 'object') {
    return mockDashboardData
  }

  return {
    summary: {
      ...mockDashboardData.summary,
      ...(payload.summary || {}),
      upcomingFixtures:
        typeof payload?.summary?.upcomingFixtures === 'number'
          ? payload.summary.upcomingFixtures
          : typeof payload?.summary?.fixturesToday === 'number'
            ? payload.summary.fixturesToday
            : mockDashboardData.summary.upcomingFixtures,
    },
    fixtures: Array.isArray(payload.fixtures) ? payload.fixtures : mockDashboardData.fixtures,
    fixturesSource:
      typeof payload.fixturesSource === 'string'
        ? payload.fixturesSource
        : mockDashboardData.fixturesSource,
    standingsSource:
      typeof payload.standingsSource === 'string'
        ? payload.standingsSource
        : mockDashboardData.standingsSource,
    standings: Array.isArray(payload.standings) ? payload.standings : mockDashboardData.standings,
    predictions: Array.isArray(payload.predictions) ? payload.predictions : mockDashboardData.predictions,
    modelMetrics:
      payload.modelMetrics && typeof payload.modelMetrics === 'object'
        ? { ...mockDashboardData.modelMetrics, ...payload.modelMetrics }
        : mockDashboardData.modelMetrics,
    dataStatus:
      payload.dataStatus && typeof payload.dataStatus === 'object'
        ? { ...mockDashboardData.dataStatus, ...payload.dataStatus }
        : mockDashboardData.dataStatus,
    liveDataStatus:
      payload.liveDataStatus && typeof payload.liveDataStatus === 'object'
        ? { ...mockDashboardData.liveDataStatus, ...payload.liveDataStatus }
        : mockDashboardData.liveDataStatus,
  }
}

export async function getDashboardData() {
  if (!API_BASE_URL) {
    return {
      data: mockDashboardData,
      source: 'mock',
    }
  }

  try {
    const response = await fetch(`${API_BASE_URL}/dashboard`)

    if (!response.ok) {
      throw new Error(`Dashboard request failed: ${response.status}`)
    }

    const payload = await response.json()

    return {
      data: normalizeDashboardData(payload),
      source: 'api',
    }
  } catch {
    return {
      data: mockDashboardData,
      source: 'mock-fallback',
    }
  }
}

export async function getTeams() {
  if (!API_BASE_URL) {
    return []
  }

  const response = await fetch(`${API_BASE_URL}/teams`)
  if (!response.ok) {
    throw new Error(`Teams request failed: ${response.status}`)
  }

  const payload = await response.json()
  return Array.isArray(payload.teams) ? payload.teams : []
}

export async function getUpcomingFixtures() {
  if (!API_BASE_URL) {
    return { fixtures: [], source: 'mock' }
  }

  const response = await fetch(`${API_BASE_URL}/fixtures/upcoming`)
  if (!response.ok) {
    throw new Error(`Upcoming fixtures request failed: ${response.status}`)
  }

  const payload = await response.json()
  return {
    fixtures: Array.isArray(payload.fixtures) ? payload.fixtures : [],
    source: typeof payload.source === 'string' ? payload.source : 'unknown',
  }
}

export async function predictMatch(homeTeam, awayTeam) {
  if (!API_BASE_URL) {
    throw new Error('API base URL is not configured')
  }

  const response = await fetch(`${API_BASE_URL}/predict-match`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ homeTeam, awayTeam }),
  })

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}))
    throw new Error(errorPayload.error || `Prediction request failed: ${response.status}`)
  }

  const payload = await response.json()
  return payload.prediction
}

export async function simulateSeason({
  iterations = 2000,
  volatility = 1.0,
  cutoffDate = '',
} = {}) {
  if (!API_BASE_URL) {
    throw new Error('API base URL is not configured')
  }

  const response = await fetch(`${API_BASE_URL}/simulate-season`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      iterations,
      volatility,
      cutoffDate: cutoffDate || null,
    }),
  })

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}))
    if (errorPayload.error === 'dataset_incomplete') {
      const teamCount = errorPayload?.details?.latestSeasonTeams
      throw new Error(
        `Dataset incomplete: latest season has ${teamCount ?? 'fewer than 20'} teams. Add full-season data.`,
      )
    }
    throw new Error(errorPayload.error || `Season simulation failed: ${response.status}`)
  }

  return response.json()
}

export async function getTeamAvailability() {
  if (!API_BASE_URL) {
    return []
  }

  const response = await fetch(`${API_BASE_URL}/team-availability`)
  if (!response.ok) {
    throw new Error(`Team availability request failed: ${response.status}`)
  }

  const payload = await response.json()
  return Array.isArray(payload.teams) ? payload.teams : []
}

export async function getModelBacktest() {
  if (!API_BASE_URL) {
    return null
  }

  const response = await fetch(`${API_BASE_URL}/model-backtest`)
  if (!response.ok) {
    throw new Error(`Model backtest request failed: ${response.status}`)
  }

  const payload = await response.json()
  return payload && typeof payload === 'object' ? payload : null
}

export async function getModelInfo() {
  if (!API_BASE_URL) {
    return null
  }

  const response = await fetch(`${API_BASE_URL}/model-info`)
  if (!response.ok) {
    throw new Error(`Model info request failed: ${response.status}`)
  }

  const payload = await response.json()
  return payload && typeof payload === 'object' ? payload : null
}

export async function updateTeamAvailability(teams) {
  if (!API_BASE_URL) {
    throw new Error('API base URL is not configured')
  }

  const response = await fetch(`${API_BASE_URL}/team-availability`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ teams }),
  })

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}))
    throw new Error(errorPayload.error || `Team availability update failed: ${response.status}`)
  }

  return response.json()
}
