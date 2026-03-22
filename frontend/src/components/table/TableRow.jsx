function TableRow({ team, index }) {
  const changeStyles = {
    up: 'text-emerald-300 bg-emerald-400/10 border-emerald-400/20',
    down: 'text-rose-300 bg-rose-400/10 border-rose-400/20',
    same: 'text-slate-300 bg-white/5 border-white/10',
  }

  return (
    <tr
      className="row-animate border-b border-white/5 text-sm text-slate-200 hover:bg-white/5"
      style={{ animationDelay: `${index * 80}ms` }}
    >
      <td className="px-3 py-3 font-semibold text-slate-100">{team.position}</td>
      <td className="px-3 py-3">{team.name}</td>
      <td className="px-3 py-3 text-center">{team.played}</td>
      <td className="px-3 py-3 text-center">{team.points}</td>
      <td className="px-3 py-3">
        <span
          className={[
            'inline-flex min-w-14 items-center justify-center rounded-full border px-2 py-1 text-xs',
            changeStyles[team.change] || changeStyles.same,
          ].join(' ')}
        >
          {team.changeLabel}
        </span>
      </td>
    </tr>
  )
}

export default TableRow
