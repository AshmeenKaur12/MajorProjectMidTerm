import React, { useState } from 'react';
import {
    Activity,
    AlertCircle,
    ArrowRight,
    CheckCircle2,
    Loader2,
    ShieldCheck,
    UserRound
} from 'lucide-react';

import './RiskAssessment.css';

const API_URL = 'http://127.0.0.1:8000';

const initialForm = {
    yrtx: '',
    prevtx: '',
    ageattx: '',
    sex: '',
    ethnicity: '',

    prd: '',
    rdm: '',
    rbmi: '',

    premt_tx: '',

    dtype: '',
    dageattx: '',

    dcmv: '',
    debv: '',

    rcmv: '',
    rebv: '',

    mma: '',
    mmb: '',
    mmdr: '',
    totalmm: '',
    abomm: '',

    crf: '',
    cit: '',

    induction: '',

    pra: '',

    age60: '',
    citcat: '',
    age10: ''
};


/* ============================================================
   FIELD COMPONENT
   ============================================================ */

function Field({
    label,
    name,
    value,
    onChange,
    type = 'text',
    placeholder,
    required = false,
    children
}) {
    return (
        <div className="form-field">
            <label htmlFor={name}>
                {label}
                {required && <span className="required">*</span>}
            </label>

            {children ? (
                <select
                    id={name}
                    name={name}
                    value={value}
                    onChange={onChange}
                    required={required}
                >
                    <option value="">Select</option>
                    {children}
                </select>
            ) : (
                <input
                    id={name}
                    name={name}
                    type={type}
                    value={value}
                    onChange={onChange}
                    placeholder={placeholder}
                    required={required}
                    min={type === 'number' ? 0 : undefined}
                    step={type === 'number' ? 'any' : undefined}
                />
            )}
        </div>
    );
}


/* ============================================================
   HELPERS
   ============================================================ */

function numberOrNull(value) {
    if (value === '' || value === null || value === undefined) {
        return null;
    }

    const number = Number(value);

    return Number.isFinite(number) ? number : null;
}


function stringOrNull(value) {
    if (value === '' || value === null || value === undefined) {
        return null;
    }

    return String(value);
}


/* ============================================================
   COMPONENT
   ============================================================ */

export default function RiskAssessment() {
    const [form, setForm] = useState(initialForm);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
const [explanation, setExplanation] = useState(null);


    /* ==========================================================
       HANDLE CHANGE
       ========================================================== */

    const handleChange = (event) => {
        const { name, value } = event.target;

        setForm((previous) => ({
            ...previous,
            [name]: value
        }));

        // Remove previous error as soon as user changes a field.
        if (error) {
            setError('');
        }
    };


    /* ==========================================================
       BUILD API PAYLOAD
       ========================================================== */

    const buildPayload = () => {
        return {
            // Numeric fields
            yrtx: numberOrNull(form.yrtx),
            prevtx: numberOrNull(form.prevtx),
            ageattx: numberOrNull(form.ageattx),

            // Categorical fields
            sex: stringOrNull(form.sex),
            ethnicity: stringOrNull(form.ethnicity),
            prd: stringOrNull(form.prd),
            rdm: stringOrNull(form.rdm),

            // Numeric
            rbmi: numberOrNull(form.rbmi),
            premt_tx: numberOrNull(form.premt_tx),

            // Donor
            dtype: stringOrNull(form.dtype),
            dageattx: numberOrNull(form.dageattx),
            dcmv: stringOrNull(form.dcmv),
            debv: stringOrNull(form.debv),

            // Recipient
            rcmv: stringOrNull(form.rcmv),
            rebv: stringOrNull(form.rebv),

            // HLA / immune
            mma: numberOrNull(form.mma),
            mmb: numberOrNull(form.mmb),
            mmdr: numberOrNull(form.mmdr),
            totalmm: numberOrNull(form.totalmm),
            abomm: numberOrNull(form.abomm),

            // Transplant
            crf: numberOrNull(form.crf),
            cit: numberOrNull(form.cit),
            induction: stringOrNull(form.induction),
            pra: numberOrNull(form.pra),

            // Derived/categorical fields
            age60: stringOrNull(form.age60),
            citcat: stringOrNull(form.citcat),

            // Keep as string because backend currently defines age10 as string.
            age10: stringOrNull(form.age10)
        };
    };


    /* ==========================================================
       VALIDATE PAYLOAD
       ========================================================== */

    const validatePayload = (payload) => {
        const requiredFields = [
            'yrtx',
            'prevtx',
            'ageattx',
            'sex',
            'ethnicity',
            'prd',
            'rdm',
            'rbmi',
            'premt_tx',
            'dtype',
            'dageattx',
            'dcmv',
            'debv',
            'rcmv',
            'rebv',
            'mma',
            'mmb',
            'mmdr',
            'totalmm',
            'abomm',
            'crf',
            'cit',
            'induction',
            'pra',
            'age60',
            'citcat',
            'age10'
        ];

        const missing = requiredFields.filter(
            (field) =>
                payload[field] === null ||
                payload[field] === undefined ||
                payload[field] === ''
        );

        if (missing.length > 0) {
            return `Please complete all required fields. Missing: ${missing.join(', ')}`;
        }

        return null;
    };


    /* ==========================================================
       SUBMIT
       ========================================================== */

    const handleSubmit = async (event) => {
        event.preventDefault();

        setLoading(true);
        setError('');
        setResult(null);

        try {
            const payload = buildPayload();

            const validationError = validatePayload(payload);

            if (validationError) {
                throw new Error(validationError);
            }

            console.log('Sending prediction payload:', payload);

            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',

                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },

                body: JSON.stringify(payload)
            });


            /* -------------------------------------------------------
               Safely parse response
            ------------------------------------------------------- */

            const contentType =
                response.headers.get('content-type') || '';

            let data;

            if (contentType.includes('application/json')) {
                data = await response.json();
            } else {
                const text = await response.text();

                throw new Error(
                    text || `Server returned HTTP ${response.status}`
                );
            }


            /* -------------------------------------------------------
               Handle FastAPI errors
            ------------------------------------------------------- */

            if (!response.ok) {
                let message = 'Prediction request failed.';

                if (typeof data?.detail === 'string') {
                    message = data.detail;
                } else if (Array.isArray(data?.detail)) {
                    message = data.detail
                        .map((item) => item.msg || 'Invalid input')
                        .join(', ');
                }

                throw new Error(message);
            }


            /* -------------------------------------------------------
               Validate prediction response
            ------------------------------------------------------- */

            if (
                typeof data.prediction !== 'number' ||
                typeof data.incompatibility_probability !== 'number' ||
                typeof data.compatibility_probability !== 'number'
            ) {
                throw new Error(
                    'The prediction API returned an invalid response.'
                );
            }

            setResult(data);

        } catch (err) {
            console.error('Prediction error:', err);

            if (err instanceof TypeError) {
                setError(
                    'Unable to connect to the prediction server. Make sure the FastAPI backend is running on port 8000.'
                );
            } else {
                setError(
                    err.message ||
                    'Unable to process the prediction request.'
                );
            }

        } finally {
            setLoading(false);
        }
    };


    /* ==========================================================
       RESET
       ========================================================== */

    const resetForm = () => {
        setForm({ ...initialForm });
        setResult(null);
        setError('');
    };


    /* ==========================================================
       FORMAT PERCENTAGE
       ========================================================== */

    const percentage = (value) => {
        if (
            typeof value !== 'number' ||
            !Number.isFinite(value)
        ) {
            return '0.0';
        }

        return (value * 100).toFixed(1);
    };


    /* ==========================================================
       RENDER
       ========================================================== */

    return (
        <section
            id="risk-assessment"
            className="assessment-section"
        >
            <div className="container">

                {/* HEADER */}

                <div className="assessment-header">

                    <div className="pill-badge">
                        <Activity size={14} />

                        <span>
                            Donor-Recipient Compatibility
                        </span>
                    </div>

                    <h2>
                        Kidney Transplant{' '}
                        <span className="text-gradient">
                            Compatibility Assessment
                        </span>
                    </h2>

                    <p>
                        Enter the available donor and recipient
                        information to generate a model-based
                        compatibility assessment.
                    </p>

                </div>


                <div className="assessment-layout">

                    {/* ==================================================
              FORM
          ================================================== */}

                    <form
                        className="assessment-card"
                        onSubmit={handleSubmit}
                    >

                        <div className="card-heading">

                            <div className="heading-icon">
                                <UserRound size={20} />
                            </div>

                            <div>
                                <h3>
                                    Patient & Transplant Information
                                </h3>

                                <p>
                                    Please provide the available
                                    information below.
                                </p>
                            </div>

                        </div>


                        {/* RECIPIENT */}

                        <div className="form-section">

                            <h4>Recipient Information</h4>

                            <div className="form-grid">

                                <Field
                                    label="Recipient Age"
                                    name="ageattx"
                                    type="number"
                                    value={form.ageattx}
                                    onChange={handleChange}
                                    placeholder="e.g. 45"
                                    required
                                />

                                <Field
                                    label="Recipient BMI"
                                    name="rbmi"
                                    type="number"
                                    value={form.rbmi}
                                    onChange={handleChange}
                                    placeholder="e.g. 24.5"
                                    required
                                />

                                <Field
                                    label="Sex"
                                    name="sex"
                                    value={form.sex}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                </Field>

                                <Field
                                    label="Ethnicity"
                                    name="ethnicity"
                                    value={form.ethnicity}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="White">White</option>
                                    <option value="Asian">Asian</option>
                                    <option value="Black">Black</option>
                                    <option value="Other">Other</option>
                                </Field>

                                <Field
                                    label="Previous Kidney Transplant"
                                    name="prevtx"
                                    value={form.prevtx}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="0">No</option>
                                    <option value="1">Yes</option>
                                </Field>

                                <Field
                                    label="Transplant Before Dialysis"
                                    name="premt_tx"
                                    value={form.premt_tx}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="0">No</option>
                                    <option value="1">Yes</option>
                                </Field>

                                <Field
                                    label="Primary Kidney Condition"
                                    name="prd"
                                    value={form.prd}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="ADPKD">ADPKD</option>
                                    <option value="Reflux/CPN">Reflux / CPN</option>
                                    <option value="Diabetes">Diabetes</option>
                                    <option value="Glomerulonephritis">
                                        Glomerulonephritis
                                    </option>
                                    <option value="Other">Other</option>
                                </Field>

                                <Field
                                    label="Relevant Kidney Disease History"
                                    name="rdm"
                                    value={form.rdm}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </Field>

                                <Field
                                    label="Age 60 or Older"
                                    name="age60"
                                    value={form.age60}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </Field>

                                <Field
                                    label="Age Group"
                                    name="age10"
                                    value={form.age10}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="0">0–9 years</option>
                                    <option value="1">10–19 years</option>
                                    <option value="2">20–29 years</option>
                                    <option value="3">30–39 years</option>
                                    <option value="4">40–49 years</option>
                                    <option value="5">50–59 years</option>
                                    <option value="6">60–69 years</option>
                                </Field>


                            </div>
                        </div>


                        {/* DONOR */}

                        <div className="form-section">

                            <h4>Donor Information</h4>

                            <div className="form-grid">

                                <Field
                                    label="Donor Age"
                                    name="dageattx"
                                    type="number"
                                    value={form.dageattx}
                                    onChange={handleChange}
                                    placeholder="e.g. 40"
                                    required
                                />

                                <Field
                                    label="Donor Type"
                                    name="dtype"
                                    value={form.dtype}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="Living donor">
                                        Living Donor
                                    </option>

                                    <option value="Deceased donor">
                                        Deceased Donor
                                    </option>
                                </Field>

                                <Field
                                    label="Donor CMV Status"
                                    name="dcmv"
                                    value={form.dcmv}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="Positive">Positive</option>
                                    <option value="Negative">Negative</option>
                                </Field>

                                <Field
                                    label="Donor EBV Status"
                                    name="debv"
                                    value={form.debv}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="Positive">Positive</option>
                                    <option value="Negative">Negative</option>
                                </Field>

                            </div>
                        </div>


                        {/* BLOOD & IMMUNE */}

                        <div className="form-section">

                            <h4>Blood & Immune Compatibility</h4>

                            <div className="form-grid">

                                <Field
                                    label="HLA Mismatch — A"
                                    name="mma"
                                    type="number"
                                    value={form.mma}
                                    onChange={handleChange}
                                    placeholder="e.g. 0"
                                    required
                                />

                                <Field
                                    label="HLA Mismatch — B"
                                    name="mmb"
                                    type="number"
                                    value={form.mmb}
                                    onChange={handleChange}
                                    placeholder="e.g. 0"
                                    required
                                />

                                <Field
                                    label="HLA Mismatch — DR"
                                    name="mmdr"
                                    type="number"
                                    value={form.mmdr}
                                    onChange={handleChange}
                                    placeholder="e.g. 0"
                                    required
                                />

                                <Field
                                    label="Total HLA Mismatch"
                                    name="totalmm"
                                    type="number"
                                    value={form.totalmm}
                                    onChange={handleChange}
                                    placeholder="e.g. 0"
                                    required
                                />

                                <Field
                                    label="Blood Group Match"
                                    name="abomm"
                                    type="number"
                                    value={form.abomm}
                                    onChange={handleChange}
                                    placeholder="e.g. 0"
                                    required
                                />

                                <Field
                                    label="PRA (Antibody Sensitization Level)"
                                    name="pra"
                                    type="number"
                                    value={form.pra}
                                    onChange={handleChange}
                                    placeholder="e.g. 0"
                                    required
                                />

                                <Field
                                    label="Recipient CMV Status"
                                    name="rcmv"
                                    value={form.rcmv}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="Positive">Positive</option>
                                    <option value="Negative">Negative</option>
                                </Field>

                                <Field
                                    label="Recipient EBV Status"
                                    name="rebv"
                                    value={form.rebv}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="Positive">Positive</option>
                                    <option value="Negative">Negative</option>
                                </Field>

                            </div>
                        </div>


                        {/* TRANSPLANT */}

                        <div className="form-section">

                            <h4>Transplant Information</h4>

                            <div className="form-grid">

                                <Field
                                    label="Cold Ischemia Time"
                                    name="cit"
                                    type="number"
                                    value={form.cit}
                                    onChange={handleChange}
                                    placeholder="e.g. 4"
                                    required
                                />

                                <Field
                                    label="Cold Ischemia Time Category"
                                    name="citcat"
                                    value={form.citcat}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="Low">Low</option>
                                    <option value="Moderate">Moderate</option>
                                    <option value="High">High</option>
                                </Field>

                                <Field
                                    label="Year of Transplant"
                                    name="yrtx"
                                    type="number"
                                    value={form.yrtx}
                                    onChange={handleChange}
                                    placeholder="e.g. 2020"
                                    required
                                />

                                <Field
                                    label="Initial Anti-Rejection Treatment"
                                    name="induction"
                                    value={form.induction}
                                    onChange={handleChange}
                                    required
                                >
                                    <option value="Campath">Campath</option>
                                    <option value="Simulect">Simulect</option>
                                </Field>

                                <Field
                                    label="Chronic Kidney Disease Indicator"
                                    name="crf"
                                    type="number"
                                    value={form.crf}
                                    onChange={handleChange}
                                    placeholder="e.g. 0"
                                    required
                                />

                            </div>
                        </div>


                        {/* ERROR */}

                        {error && (
                            <div className="assessment-error">
                                <AlertCircle size={18} />
                                <span>{error}</span>
                            </div>
                        )}


                        {/* ACTIONS */}

                        <div className="assessment-actions">

                            <button
                                type="submit"
                                className="btn-primary assessment-submit"
                                disabled={loading}
                            >
                                {loading ? (
                                    <>
                                        <Loader2
                                            size={18}
                                            className="spin"
                                        />

                                        Analysing Information...
                                    </>
                                ) : (
                                    <>
                                        Assess Compatibility
                                        <ArrowRight size={17} />
                                    </>
                                )}
                            </button>

                            <button
                                type="button"
                                className="btn-reset"
                                onClick={resetForm}
                                disabled={loading}
                            >
                                Clear Form
                            </button>

                        </div>

                    </form>


                    {/* ==================================================
              RESULTS
          ================================================== */}

                    <div className="results-column">

                        {!result && !loading && (
                            <div className="result-placeholder">

                                <div className="placeholder-icon">
                                    <ShieldCheck size={34} />
                                </div>

                                <h3>Assessment Results</h3>

                                <p>
                                    Complete the assessment to view
                                    the model-generated compatibility result.
                                </p>

                            </div>
                        )}


                        {loading && (
                            <div className="result-placeholder">

                                <Loader2
                                    size={36}
                                    className="spin medical-spinner"
                                />

                                <h3>Analysing Information</h3>

                                <p>
                                    The prediction model is processing
                                    the donor and recipient information.
                                </p>

                            </div>
                        )}


                        {result && (
                            <div className="result-card">

                                <div className="result-card-header">

                                    <div>
                                        <span className="result-label">
                                            MODEL ASSESSMENT
                                        </span>

                                        <h3>
                                            Compatibility Result
                                        </h3>
                                    </div>

                                    {result.prediction === 0 ? (
                                        <div className="result-status compatible">
                                            <CheckCircle2 size={20} />
                                            Compatible
                                        </div>
                                    ) : (
                                        <div className="result-status incompatible">
                                            <AlertCircle size={20} />
                                            Incompatible
                                        </div>
                                    )}

                                </div>


                                {/* PROBABILITY */}

                                <div className="probability-section">

                                    <div
                                        className="probability-ring"
                                        style={{
                                            '--progress': `${Math.min(
                                                Math.max(
                                                    result.incompatibility_probability * 100,
                                                    0
                                                ),
                                                100
                                            )}%`
                                        }}
                                    >
                                        <div className="probability-ring-content">
                                            <strong>
                                                {percentage(
                                                result.incompatibility_probability
                                            )}
                                            %
                                            </strong>

                                            <span>
                                                predicted risk
                                            </span>
                                        </div>
                                    </div>



                                    <div className="probability-copy">

                                        <span>
                                            Predicted Incompatibility Risk
                                        </span>

                                        <strong>
                                            {percentage(
                                                result.incompatibility_probability
                                            )}
                                            %
                                        </strong>



                                    </div>

                                </div>


                                {/* RISK MESSAGE */}

                                <div className="risk-message">

                                    <Activity size={20} />

                                    <div>

                                        <strong>
                                            {result.risk_category}
                                        </strong>

                                        <p>
                                            This result is generated by the
                                            trained machine-learning model using
                                            the information provided.
                                        </p>

                                    </div>

                                </div>


                                {/* RESULT SUMMARY */}

                                <div className="result-grid">

                                    <div>
                                        <span>
                                            Predicted&nbsp;&nbsp;&nbsp; Compatibility
                                        </span>

                                        <strong>
                                            {percentage(
                                                result.compatibility_probability
                                            )}
                                            %
                                        </strong>
                                    </div>


                                    <div>
                                        <span>
                                            Predicted Incompatibility
                                        </span>

                                        <strong>
                                            {percentage(
                                                result.incompatibility_probability
                                            )}
                                            %
                                        </strong>
                                    </div>




                                </div>




                            </div>
                        )}

                    </div>

                </div>
            </div>
        </section>
    );
}