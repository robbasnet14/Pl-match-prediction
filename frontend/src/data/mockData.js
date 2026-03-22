export const dashboardSummary = {
  upcomingFixtures: 5,
  predictionConfidence: 78,
  updatedAt: '2026-02-20T12:00:00Z',
  activeModels: 3,
}

export const navSections = ['Dashboard', 'Match Prediction', 'Season Simulation']

export const upcomingFixtures = [
  { id: 1, home: 'Arsenal', away: 'Brighton', time: '13:30', venue: 'Emirates' },
  { id: 2, home: 'Liverpool', away: 'Newcastle', time: '16:00', venue: 'Anfield' },
  { id: 3, home: 'Tottenham', away: 'Chelsea', time: '18:30', venue: 'Tottenham Hotspur Stadium' },
]

export const leagueTable = [
  { position: 1, name: 'Liverpool', played: 25, points: 57, change: 'same', changeLabel: '-' },
  { position: 2, name: 'Arsenal', played: 25, points: 55, change: 'up', changeLabel: '+1' },
  { position: 3, name: 'Manchester City', played: 25, points: 53, change: 'down', changeLabel: '-1' },
  { position: 4, name: 'Aston Villa', played: 25, points: 48, change: 'same', changeLabel: '-' },
  { position: 5, name: 'Tottenham', played: 25, points: 46, change: 'up', changeLabel: '+1' },
  { position: 6, name: 'Manchester United', played: 25, points: 42, change: 'down', changeLabel: '-1' },
]

export const predictionHighlights = [
  {
    id: 1,
    fixture: 'Arsenal vs Brighton',
    home: 'Arsenal',
    away: 'Brighton',
    confidence: 81,
    score: '2-1',
    probabilities: { home: 62, draw: 22, away: 16 },
    expectedGoals: { home: 1.86, away: 1.14, total: 3.0 },
    topScorelines: [
      { score: '2-1', probabilityPct: 17.2 },
      { score: '2-0', probabilityPct: 13.8 },
      { score: '1-1', probabilityPct: 11.1 },
      { score: '3-1', probabilityPct: 8.4 },
      { score: '3-0', probabilityPct: 7.1 },
      { score: '2-2', probabilityPct: 6.2 },
    ],
    goalOutlook: {
      total3PlusPct: 58.4,
      total4PlusPct: 34.7,
      bttsPct: 56.1,
      home3PlusPct: 39.5,
      away3PlusPct: 17.2,
    },
    goalMarkets: {
      over25Pct: 58.4,
      under25Pct: 41.6,
      over35Pct: 34.7,
      under35Pct: 65.3,
      bttsPct: 56.1,
      noBttsPct: 43.9,
    },
  },
  {
    id: 2,
    fixture: 'Liverpool vs Newcastle',
    home: 'Liverpool',
    away: 'Newcastle',
    confidence: 76,
    score: '2-0',
    probabilities: { home: 58, draw: 24, away: 18 },
    expectedGoals: { home: 1.72, away: 0.94, total: 2.66 },
    topScorelines: [
      { score: '2-0', probabilityPct: 16.3 },
      { score: '2-1', probabilityPct: 12.9 },
      { score: '1-0', probabilityPct: 11.8 },
      { score: '3-1', probabilityPct: 8.1 },
      { score: '3-0', probabilityPct: 7.9 },
      { score: '1-1', probabilityPct: 6.7 },
    ],
    goalOutlook: {
      total3PlusPct: 54.2,
      total4PlusPct: 30.6,
      bttsPct: 49.3,
      home3PlusPct: 35.6,
      away3PlusPct: 14.8,
    },
    goalMarkets: {
      over25Pct: 54.2,
      under25Pct: 45.8,
      over35Pct: 30.6,
      under35Pct: 69.4,
      bttsPct: 49.3,
      noBttsPct: 50.7,
    },
  },
]

export const mockDashboardData = {
  summary: dashboardSummary,
  fixtures: upcomingFixtures,
  fixturesSource: 'mock',
  standingsSource: 'mock',
  standings: leagueTable,
  predictions: predictionHighlights,
  modelMetrics: {
    accuracyPct: 68,
    evaluationScope: 'season_holdout',
    sampleCount: 1200,
    lastTrainedAt: '2026-02-20T13:00:00Z',
  },
  modelBacktest: {
    available: false,
    reason: 'mock_data',
    accuracyPct: 0,
    brierScore: 0,
    homeGoalMae: 0,
    awayGoalMae: 0,
    scorelineExactPct: 0,
    confidenceCalibration: [],
  },
  dataStatus: {
    isComplete: false,
    latestSeason: 2022,
    latestSeasonTeams: 19,
    latestMatchDate: '2022-05-22',
    seasonTeamCounts: [{ season: 2022, teamCount: 19, matchCount: 694 }],
    issues: ['latest_season_has_fewer_than_20_teams'],
  },
  liveDataStatus: {
    configured: false,
    fixturesUsingExternal: false,
    standingsUsingExternal: false,
    message: 'Live API key missing. Showing fallback data for fixtures and standings.',
  },
}
