import { NavLink } from 'react-router-dom'

const navItems = [
  { label: 'Dashboard', to: '/dashboard' },
  { label: 'Match Prediction', to: '/match-prediction' },
  { label: 'Season Simulation', to: '/season-simulation' },
  { label: 'Team Availability', to: '/team-availability' },
]

function Sidebar() {
  return (
    <aside className="panel-glass hidden w-72 flex-col border-r border-white/10 p-5 lg:flex">
      <div className="mb-8">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Premier League</p>
        <h1 className="mt-2 text-2xl font-semibold text-white">Match Intelligence</h1>
      </div>

      <nav className="space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              [
                'block rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200',
                isActive
                  ? 'bg-indigo-500/20 text-indigo-200 ring-1 ring-indigo-400/40'
                  : 'text-slate-300 hover:bg-white/10 hover:text-white',
              ].join(' ')
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}

export default Sidebar
