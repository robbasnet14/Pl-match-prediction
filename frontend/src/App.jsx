import { Navigate, Route, Routes } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard'
import MatchPrediction from './pages/MatchPrediction'
import SeasonSimulation from './pages/SeasonSimulation'
import TeamAvailability from './pages/TeamAvailability'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="match-prediction" element={<MatchPrediction />} />
        <Route path="season-simulation" element={<SeasonSimulation />} />
        <Route path="team-availability" element={<TeamAvailability />} />
      </Route>
    </Routes>
  )
}

export default App
