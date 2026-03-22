import TableRow from './TableRow'

function LeagueTable({ teams }) {
  return (
    <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/20">
      <table className="w-full border-collapse">
        <thead>
          <tr className="bg-white/5 text-left text-xs uppercase tracking-[0.14em] text-slate-400">
            <th className="px-3 py-3">#</th>
            <th className="px-3 py-3">Team</th>
            <th className="px-3 py-3 text-center">P</th>
            <th className="px-3 py-3 text-center">Pts</th>
            <th className="px-3 py-3">Move</th>
          </tr>
        </thead>
        <tbody>
          {teams.map((team, index) => (
            <TableRow key={team.name} team={team} index={index} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default LeagueTable
