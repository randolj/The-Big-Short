import { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
  Link,
  useParams,
} from "react-router-dom";
import "./App.css";

function App() {
  useEffect(() => {
    fetch("http://localhost:8000/addresses")
      .then(res => res.json())
      .then(data => console.log("Addresses from backend:", data))
      .catch(err => console.error("Fetch error:", err));
  }, []);
  
  return (
    <Router>
      <div className="app-container">
        <nav className="navbar">
        <Link to="/" className="logo nav-logo" style={{ paddingLeft: 0 }}>
          The Big Short
        </Link>
          <div className="nav-links">
            <Link to="/" className="nav-link">
              Search
            </Link>
            <Link to="/custom" className="nav-link">
              Value My Home
            </Link>
            {/* <Link to="/about" className="nav-link">
              About
            </Link> */}
          </div>
        </nav>
        <div className="content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/zip/:zipcode" element={<ZipResults />} />
            <Route path="/address/:address" element={<AddressResults />} />
            <Route path="/custom" element={<CustomValuation />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

function Home() {
  const [input, setInput] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    if (input.length > 2 && !/^\d{5}$/.test(input)) {
      fetch(`http://localhost:8000/search-addresses?query=${input}`)
        .then((res) => res.json())
        .then((data) => setSuggestions(data))
        .catch((err) => console.error(err));
    } else {
      setSuggestions([]);
    }
  }, [input]);

  const handleSearch = async (value) => {
    const isZip = /^\d{5}$/.test(value);
  
    if (isZip) {
      navigate(`/zip/${value}`);
    } else {
      // Navigate to address route
      navigate(`/address/${encodeURIComponent(value)}`);
    }
  
    setInput(value);
    setSuggestions([]);
  };
  
  const [randomHomes, setRandomHomes] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/random-properties")
      .then(res => res.json())
      .then(data => setRandomHomes(data.properties))
      .catch(err => console.error("Error fetching random homes:", err));
  }, []);


  return (
    <>
      <div className="hero-section">
        <h1 className="hero-title">Find Your Dream Home's True Value</h1>
        <p className="hero-subtitle">Advanced AI-powered valuations that see beyond the listing price</p>
        <div className="search-container" style={{ position: "relative" }}>
          <div className="search-box">
            <div className="search-icon">📍</div>
            <input
              className="search-input"
              placeholder="Enter an address or ZIP code"
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button className="search-button" onClick={() => handleSearch(input)}>
              Search <span>→</span>
            </button>
          </div>
          {suggestions.length > 0 && (
            <ul className="autocomplete-list">
              {suggestions.map((addr, idx) => (
                <li key={idx} onClick={() => handleSearch(addr)}>
                  {addr}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="features-section">
        <h2 className="section-heading">Why The Big Short is Different</h2>
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">📈</div>
            <h3 className="feature-title">Predictive Analytics</h3>
            <p className="feature-description">
              Our AI model analyzes thousands of market factors to predict future value trends.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🏠</div>
            <h3 className="feature-title">Custom Valuations</h3>
            <p className="feature-description">
              Tailor your property's valuation with specific features and details.
            </p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">💰</div>
            <h3 className="feature-title">Market Insights</h3>
            <p className="feature-description">
              Get deep market analysis with trends and comparative property data.
            </p>
          </div>
        </div>
      </div>

      <div className="results-container">
        <h2 className="section-heading">Recently Valued Properties</h2>
        <div className="results-grid">
        {randomHomes.map((home, i) => (
          <Link
            to={`/address/${encodeURIComponent(home.address)}`}
            key={i}
            className="property-card"
            style={{ textDecoration: "none", color: "inherit" }}
          >
            <div className="property-info">
              <h3 className="property-address">{home.address}</h3>
              <div className="property-location">
                <span>{home.city}, FL {home.zip}</span>
                <span className={`property-tag ${home.predicted < home.current ? 'down' : ''}`}>
                  {home.predicted > home.current ? "Trending Up" : home.predicted < home.current ? "Trending Down" : "Stable"}
                </span>
              </div>
              <div className="property-values">
                <div className="value-item">
                  <div className="value-label">Current Value</div>
                  <div className="current-value">${Math.round(home.current).toLocaleString()}</div>
                </div>
                <div className="value-item">
                  <div className="value-label">Predicted Value</div>
                  <div className="predicted-value">${Math.round(home.predicted).toLocaleString()}</div>
                </div>
              </div>
            </div>
          </Link>
        ))}
        </div>
      </div>
    </>
  );
}

function ZipResults() {
  const { zipcode } = useParams();
  const [properties, setProperties] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`http://localhost:8000/zip-properties?zipcode=${zipcode}`)
      .then(res => res.json())
      .then(data => {
        setProperties(data.properties);
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching ZIP properties:", err);
        setLoading(false);
      });
  }, [zipcode]);

  if (loading) return <div className="container">Loading properties in ZIP {zipcode}...</div>;

  if (properties.length === 0) return <div className="container">No properties found in ZIP {zipcode}.</div>;

  return (
    <div className="container">
      <h2 className="section-title">Properties in ZIP {zipcode}</h2>
      <div className="results-grid">
        {properties.map((p, i) => (
          <Link
            to={`/address/${encodeURIComponent(p.address)}`}
            key={i}
            className="property-card"
            style={{ textDecoration: "none", color: "inherit" }}
          >
            <div className="property-info">
              <h3 className="property-address">{p.address}</h3>
              <div className="property-location">
                <span>ZIP: {zipcode}</span>
                <span className={`property-tag ${p.predicted < p.current ? 'down' : ''}`}>
                  {p.predicted > p.current ? "Trending Up" : p.predicted < p.current ? "Trending Down" : "Stable"}
                </span>
              </div>
              <div className="property-values">
                <div className="value-item">
                  <div className="value-label">Current Value</div>
                  <div className="current-value">${Math.round(p.current).toLocaleString()}</div>
                </div>
                <div className="value-item">
                  <div className="value-label">Predicted Value</div>
                  <div className="predicted-value">${Math.round(p.predicted).toLocaleString()}</div>
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}

function AddressResults() {
  const { address } = useParams();
  const [details, setDetails] = useState(null);
  const [estimatedValue, setEstimatedValue] = useState(null);
  const [featureInputs, setFeatureInputs] = useState(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch(`http://localhost:8000/address-details?address=${encodeURIComponent(address)}`);
        const data = await res.json();
        setDetails(data);

        const predRes = await fetch(`http://localhost:8000/predict-by-address?address=${encodeURIComponent(address)}`);
        const predData = await predRes.json();
        setEstimatedValue(predData.predicted_value);
        setFeatureInputs(predData.features);
      } catch (err) {
        console.error("Error:", err);
      }
    }

    fetchData();
  }, [address]);

  if (!details) return <div className="container">Loading address info...</div>;

  return (
    <div className="property-detail-card">
      <h2 className="section-title">{details.PropertyAddressFull}</h2>
      <div className="house-info">
        <p><strong>City:</strong> {details.PropertyAddressCity}</p>
        <p><strong>ZIP:</strong> {details.PropertyAddressZIP}</p>
        <p><strong>Assessed Value:</strong> ${Number(details.TaxMarketValueTotal).toLocaleString()}</p>

        {featureInputs && (
          <>
            <h3 className="section-title">Key Home Attributes</h3>
            <div className="key-attributes-grid">
              <div className="key-attribute"><strong>Year Built:</strong> {featureInputs.YearBuilt}</div>
              <div className="key-attribute"><strong>Pool:</strong> {featureInputs.Pool === 1 ? "Yes" : "No"}</div>
              <div className="key-attribute"><strong>Lot Size:</strong> {Number(featureInputs.AreaLotSF).toLocaleString()} sq ft</div>
              <div className="key-attribute"><strong>Bathrooms:</strong> {featureInputs.BathCount}</div>
              <div className="key-attribute"><strong>Bedrooms:</strong> {featureInputs.BedroomsCount}</div>
              <div className="key-attribute"><strong>Stories:</strong> {featureInputs.StoriesCount}</div>
            </div>
          </>
        )}

        {estimatedValue && (
          <div className="predicted-banner">
            Predicted Market Value: <strong>${Number(estimatedValue).toLocaleString()}</strong>
          </div>
        )}
      </div>
    </div>
  );
}


function CustomValuation() {
  const [form, setForm] = useState({
    size: "",
    bedrooms: "",
    bathrooms: "",
    yearBuilt: "",
    stories: "",
    basement: "",
    hotWaterHeating: "",
    airConditioning: "",
    mainroad: "",
    frontage: "",
    depth: "",
    backyardSize: "",
    garage: "",
    distanceFromOcean: "",
  });
  const [estimatedValue, setEstimatedValue] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    const formattedForm = Object.fromEntries(
      Object.entries(form).map(([k, v]) => [k, v === "" ? null : isNaN(v) ? v : +v])
    );
  
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(formattedForm)
      });
  
      const data = await response.json();
  
      if (data.error) {
        setEstimatedValue("Error calculating value.");
      } else {
        setEstimatedValue(`$${Number(data.predicted_value).toLocaleString()}`);
      }
    } catch (error) {
      console.error("Prediction error:", error);
      setEstimatedValue("Error calculating value.");
    }
  };
  

  return (
    <div>
      <h2 className="section-title">Value My Home</h2>
      <form onSubmit={handleSubmit} className="custom-form">
        {Object.entries(form).map(([key, val]) => (
          <div key={key} className="form-field">
            <label>{key}</label>
            <input
              type="text"
              name={key}
              value={val}
              onChange={handleChange}
            />
          </div>
        ))}
        <div className="form-button-wrapper">
          <button type="submit" className="search-button">Estimate Value</button>
        </div>
      </form>
      {estimatedValue && (
        <div className="result">Estimated Value: <strong>{estimatedValue}</strong></div>
      )}
    </div>
  );
}

function About() {
  return (
    <div>
      <h2 className="section-title">About Us & Our Model</h2>
      <p className="about-text">
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
        Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
        Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
      </p>
    </div>
  );
}

export default App;