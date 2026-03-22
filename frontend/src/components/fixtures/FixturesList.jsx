import FixtureCard from './FixtureCard'

function FixturesList({ fixtures }) {
  return (
    <div className="space-y-3">
      {fixtures.map((fixture) => (
        <FixtureCard key={fixture.id} fixture={fixture} />
      ))}
    </div>
  )
}

export default FixturesList
