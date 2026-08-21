import PredictionForm from "../components/PredictionForm";

export default function HomePage() {
  return (
    <div className="page">
      <section className="hero">
        <div className="hero__content">
          <p className="hero__eyebrow">Property valuation</p>
          <h1>What's a home like this actually worth?</h1>
          <p>
            Enter a property's details below and get an instant price estimate from a model
            trained on real listing data across major Indian cities.
          </p>
        </div>
        {/* <div className="hero__figure">
          <div>
            <p className="hero__figure-label">Model</p>
            <p className="hero__figure-value">Random forest</p>
          </div>
          <div>
            <p className="hero__figure-label">Trained on</p>
            <p className="hero__figure-value">174K+ real listings</p>
          </div>
        </div> */}
      </section>

      <PredictionForm />
    </div>
  );
}
