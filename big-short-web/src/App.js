import { useState, useEffect} from "react";
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
          <h1 className="logo">The Big Short</h1>
          <div className="nav-links">
            <Link to="/" className="nav-link">Search</Link>
            <Link to="/custom" className="nav-link">Value My Home</Link>
            <Link to="/about" className="nav-link">About</Link>
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
  

  return (
    <div className="search-box">
  <div className="search-field-wrapper">
    <input
      className="search-input"
      placeholder="Search either zip code or address"
      value={input}
      onChange={(e) => setInput(e.target.value)}
    />
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
  <button className="search-button" onClick={() => handleSearch(input)}>
    🔍 Search
  </button>
</div>
  );
}


function ZipResults() {
  const properties = [
    { address: "123 Oak Street", current: 150000, predicted: 160000 },
    { address: "456 Maple Avenue", current: 415000, predicted: 460000 },
    { address: "789 Pine Lane", current: 289000, predicted: 310000 },
    { address: "101 Birch Drive", current: 510000, predicted: 550000 },
    { address: "555 Elm Boulevard", current: 460000, predicted: 420000 },
  ];

  return (
    <div>
      <h2 className="section-title">List of Properties</h2>
      <table className="results-table">
        <thead>
          <tr>
            <th>Address</th>
            <th>Current Value</th>
            <th>Predicted Value</th>
            <th>Trend</th>
          </tr>
        </thead>
        <tbody>
          {properties.map((p, i) => (
            <tr key={i}>
              <td>{p.address}</td>
              <td>${p.current.toLocaleString()}</td>
              <td>${p.predicted.toLocaleString()}</td>
              <td>
                {p.predicted > p.current ? "🔼" : p.predicted < p.current ? "🔽" : "⏺"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AddressResults() {
  const { address } = useParams();
  const [details, setDetails] = useState(null);

  useEffect(() => {
    fetch(`http://localhost:8000/address-details?address=${encodeURIComponent(address)}`)
      .then(res => res.json())
      .then(data => setDetails(data))
      .catch(err => console.error("Error fetching address details:", err));
  }, [address]);

  if (!details) return <div>Loading address info...</div>;

  return (
    <div>
      <h2 className="section-title">{details.PropertyAddressFull}</h2>
      <div className="house-info">
        <strong>City:</strong> {details.PropertyAddressCity} <br />
        <strong>ZIP:</strong> {details.PropertyAddressZIP} <br />
        <strong>Year Built:</strong> {details.YearBuilt} <br />
        <strong>Market Value:</strong> ${Number(details.TaxMarketValueTotal).toLocaleString()} <br />
        {/* Add more fields here as needed */}
      </div>
    </div>
  );
}


function CustomValuation() {
  const [form, setForm] = useState({
    size: "",
    bedrooms: "",
    bathrooms: "",
    age: "",
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

  const handleSubmit = (e) => {
    e.preventDefault();
    // Replace this with an API call to your model backend
    setEstimatedValue("$350,000");
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