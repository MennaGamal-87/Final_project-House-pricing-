import { Link } from "react-router-dom";

export default function NotFoundPage() {
  return (
    <div className="page not-found">
      <h1>404</h1>
      <p style={{ color: "var(--ink-soft)", marginBottom: 24 }}>
        That page doesn't exist.
      </p>
      <Link to="/" className="btn-secondary">
        Back to the estimator
      </Link>
    </div>
  );
}
