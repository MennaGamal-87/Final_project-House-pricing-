import { Link, Navigate, useLocation } from "react-router-dom";
import type { PredictionRequest, PredictionResponse } from "../types/prediction";

interface LocationState {
  prediction: PredictionResponse;
  form: PredictionRequest;
}

function formatLocationLabel(slug: string): string {
  if (slug === "other") return "Other / not listed";
  return slug
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatIndianPrice(amount: number): string {
  if (amount >= 1e7) return `₹ ${(amount / 1e7).toFixed(2)} Cr`;
  if (amount >= 1e5) return `₹ ${(amount / 1e5).toFixed(2)} Lac`;
  return `₹ ${Math.round(amount).toLocaleString("en-IN")}`;
}

export default function ResultPage() {
  const location = useLocation();
  const state = location.state as LocationState | null;

  if (!state?.prediction) {
    return <Navigate to="/" replace />;
  }

  const { prediction, form } = state;

  return (
    <div className="page result-page">
      <div className="card">
        <p className="result-eyebrow">Estimated price</p>
        <p className="result-price">{formatIndianPrice(prediction.predicted_price)}</p>
        <p className="result-summary">
          Based on a <strong>{form.carpet_area_sqft.toLocaleString("en-IN")} sqft</strong>{" "}
          property in <strong>{formatLocationLabel(form.location)}</strong>, floor{" "}
          {form.floor_num}, with{" "}
          {form.bathroom} bathroom{form.bathroom === 1 ? "" : "s"} and {form.balcony} balcon
          {form.balcony === 1 ? "y" : "ies"}.
        </p>

        <div className="summary-grid">
          <div className="summary-item">
            <span className="summary-item__label">Furnishing</span>
            <span className="summary-item__value">{form.furnishing}</span>
          </div>
          <div className="summary-item">
            <span className="summary-item__label">Transaction</span>
            <span className="summary-item__value">{form.transaction}</span>
          </div>
          <div className="summary-item">
            <span className="summary-item__label">Ownership</span>
            <span className="summary-item__value">{form.ownership}</span>
          </div>
          <div className="summary-item">
            <span className="summary-item__label">Facing</span>
            <span className="summary-item__value">{form.facing}</span>
          </div>
          <div className="summary-item">
            <span className="summary-item__label">Carpet area</span>
            <span className="summary-item__value">{form.carpet_area_sqft} sqft</span>
          </div>
          <div className="summary-item">
            <span className="summary-item__label">Floor</span>
            <span className="summary-item__value">{form.floor_num}</span>
          </div>
        </div>

        <Link to="/" className="btn-secondary">
          Estimate another property
        </Link>
      </div>
    </div>
  );
}
