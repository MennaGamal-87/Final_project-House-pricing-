import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import {
  FACING_OPTIONS,
  FURNISHING_OPTIONS,
  OWNERSHIP_OPTIONS,
  TRANSACTION_OPTIONS,
  type PredictionRequest,
} from "../types/prediction";
import { ApiError, predictPrice } from "../api/predictionClient";
import locations from "../locations.json";

type FormState = {
  location: string;
  carpet_area_sqft: string;
  floor_num: string;
  bathroom: string;
  balcony: string;
  furnishing: string;
  transaction: string;
  ownership: string;
  facing: string;
};

const initialState: FormState = {
  location: "",
  carpet_area_sqft: "",
  floor_num: "",
  bathroom: "",
  balcony: "",
  furnishing: FURNISHING_OPTIONS[0],
  transaction: TRANSACTION_OPTIONS[0],
  ownership: OWNERSHIP_OPTIONS[0],
  facing: FACING_OPTIONS[0],
};

const allLocations: string[] = Array.isArray(locations) ? locations : [];

// Backend locations are lowercase slugs (e.g. "new-delhi", "greater-noida") straight from
// training - we send that exact value to the API, but show a friendlier label in the UI.
function formatLocationLabel(slug: string): string {
  if (slug === "other") return "Other / not listed";
  return slug
    .split("-")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export default function PredictionForm() {
  const navigate = useNavigate();
  const [form, setForm] = useState<FormState>(initialState);
  const [errors, setErrors] = useState<Partial<Record<keyof FormState, string>>>({});
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function update<K extends keyof FormState>(key: K, value: string) {
    setForm((f) => ({ ...f, [key]: value }));
    if (errors[key]) setErrors((e) => ({ ...e, [key]: undefined }));
  }

  function validate(): boolean {
    const next: Partial<Record<keyof FormState, string>> = {};

    if (!form.location.trim()) next.location = "Choose a location.";
    if (form.carpet_area_sqft.trim() === "" || Number(form.carpet_area_sqft) <= 0) {
      next.carpet_area_sqft = "Enter an area greater than 0.";
    }
    if (form.floor_num.trim() === "" || Number.isNaN(Number(form.floor_num))) {
      next.floor_num = "Enter the floor number (0 for ground).";
    }
    if (form.bathroom.trim() === "" || Number(form.bathroom) < 0) {
      next.bathroom = "Enter the number of bathrooms.";
    }
    if (form.balcony.trim() === "" || Number(form.balcony) < 0) {
      next.balcony = "Enter the number of balconies.";
    }

    setErrors(next);
    return Object.keys(next).length === 0;
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitError(null);

    if (!validate()) return;

    const payload: PredictionRequest = {
      location: form.location,
      carpet_area_sqft: Number(form.carpet_area_sqft),
      floor_num: Number(form.floor_num),
      bathroom: Number(form.bathroom),
      balcony: Number(form.balcony),
      furnishing: form.furnishing,
      transaction: form.transaction,
      ownership: form.ownership,
      facing: form.facing,
    };

    setLoading(true);
    try {
      const result = await predictPrice(payload);
      navigate("/result", { state: { prediction: result, form: payload } });
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Something went wrong. Please try again.";
      setSubmitError(message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="card" onSubmit={handleSubmit} noValidate>
      <div className="form-grid">
        <div className="field field--full">
          <label htmlFor="location">Location</label>
          <select
            id="location"
            value={form.location}
            onChange={(e) => update("location", e.target.value)}
            className={errors.location ? "has-error" : ""}
          >
            <option value="">Select a city</option>
            {allLocations
              .filter((loc) => loc !== "other")
              .map((loc) => (
                <option key={loc} value={loc}>
                  {formatLocationLabel(loc)}
                </option>
              ))}
          </select>
          {errors.location && <span className="field__error">{errors.location}</span>}
        </div>

        <div className="field">
          <label htmlFor="carpet_area_sqft">Carpet area (sqft)</label>
          <input
            id="carpet_area_sqft"
            type="number"
            min="1"
            placeholder="1200"
            value={form.carpet_area_sqft}
            onChange={(e) => update("carpet_area_sqft", e.target.value)}
            className={errors.carpet_area_sqft ? "has-error" : ""}
          />
          {errors.carpet_area_sqft && (
            <span className="field__error">{errors.carpet_area_sqft}</span>
          )}
        </div>

        <div className="field">
          <label htmlFor="floor_num">Floor (0 = ground)</label>
          <input
            id="floor_num"
            type="number"
            placeholder="3"
            value={form.floor_num}
            onChange={(e) => update("floor_num", e.target.value)}
            className={errors.floor_num ? "has-error" : ""}
          />
          {errors.floor_num && <span className="field__error">{errors.floor_num}</span>}
        </div>

        <div className="field">
          <label htmlFor="bathroom">Bathrooms</label>
          <input
            id="bathroom"
            type="number"
            min="0"
            placeholder="2"
            value={form.bathroom}
            onChange={(e) => update("bathroom", e.target.value)}
            className={errors.bathroom ? "has-error" : ""}
          />
          {errors.bathroom && <span className="field__error">{errors.bathroom}</span>}
        </div>

        <div className="field">
          <label htmlFor="balcony">Balconies</label>
          <input
            id="balcony"
            type="number"
            min="0"
            placeholder="1"
            value={form.balcony}
            onChange={(e) => update("balcony", e.target.value)}
            className={errors.balcony ? "has-error" : ""}
          />
          {errors.balcony && <span className="field__error">{errors.balcony}</span>}
        </div>

        <div className="field">
          <label htmlFor="furnishing">Furnishing</label>
          <select
            id="furnishing"
            value={form.furnishing}
            onChange={(e) => update("furnishing", e.target.value)}
          >
            {FURNISHING_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="transaction">Transaction type</label>
          <select
            id="transaction"
            value={form.transaction}
            onChange={(e) => update("transaction", e.target.value)}
          >
            {TRANSACTION_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="ownership">Ownership</label>
          <select
            id="ownership"
            value={form.ownership}
            onChange={(e) => update("ownership", e.target.value)}
          >
            {OWNERSHIP_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="facing">Facing</label>
          <select
            id="facing"
            value={form.facing}
            onChange={(e) => update("facing", e.target.value)}
          >
            {FACING_OPTIONS.map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="form-actions">
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading && <span className="spinner" aria-hidden="true" />}
          {loading ? "Estimating price..." : "Estimate price"}
        </button>
      </div>

      {submitError && <div className="form-error-banner">{submitError}</div>}
    </form>
  );
}
