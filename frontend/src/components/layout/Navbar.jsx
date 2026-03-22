import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { getApiHealth } from '../../services/api'

const navItems = [
  { label: 'Dashboard', to: '/dashboard' },
  { label: 'Prediction', to: '/match-prediction' },
  { label: 'Simulation', to: '/season-simulation' },
  { label: 'Availability', to: '/team-availability' },
]

function Navbar() {
  const [isApiOnline, setIsApiOnline] = useState(false)
  const [lastHealthCheck, setLastHealthCheck] = useState(null)

  useEffect(() => {
    let mounted = true

    async function checkHealth() {
      const health = await getApiHealth()
      if (!mounted) return
      setIsApiOnline(Boolean(health.ok))
      setLastHealthCheck(new Date().toISOString())
    }

    checkHealth()
    const intervalId = window.setInterval(checkHealth, 30000)

    return () => {
      mounted = false
      window.clearInterval(intervalId)
    }
  }, [])

  return (
    <header className="panel-glass sticky top-0 z-20 border-b border-white/10 px-4 py-4 sm:px-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Dashboard</p>
          <h2 className="text-lg font-semibold text-white sm:text-xl">Premier League Control Center</h2>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={[
              'rounded-full border px-3 py-1 text-xs font-medium',
              isApiOnline
                ? 'border-emerald-400/40 bg-emerald-500/10 text-emerald-300'
                : 'border-rose-400/40 bg-rose-500/10 text-rose-300',
            ].join(' ')}
            title={
              lastHealthCheck
                ? `Last check: ${new Date(lastHealthCheck).toLocaleTimeString()}`
                : 'Health check pending'
            }
          >
            {isApiOnline ? 'API Online' : 'API Offline'}
          </span>
        </div>
      </div>
      <nav className="mt-4 flex gap-2 overflow-x-auto lg:hidden">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              [
                'whitespace-nowrap rounded-lg px-3 py-2 text-xs font-medium transition-colors',
                isActive
                  ? 'bg-indigo-500/25 text-indigo-100 ring-1 ring-indigo-400/40'
                  : 'bg-white/5 text-slate-300 hover:bg-white/10',
              ].join(' ')
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </header>
  )
}

export default Navbar
