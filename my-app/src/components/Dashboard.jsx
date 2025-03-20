import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useState } from 'react';

const Dashboard = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('userEmail');
    navigate('/login');
  };

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        navigate('/login');
        return;
      }
      const res = await axios.post(
        // 'http://localhost:8001/query',
        '/query',
        { question: query },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      console.log('Response from backend:', res.data); // Debug the response
      setResponse(res.data.response || 'No response content')
    } catch (err) {
      setResponse('Error: Could not process query.');
      console.error(err);
    }
  };

  return (
    <div className="dashboard">
      <h2>Dashboard</h2>
      <p>Welcome, {localStorage.getItem('userEmail')}!</p>
      <button onClick={handleLogout}>Logout</button>
      <div>
        <h3>Submit a Query</h3>
        <form onSubmit={handleQuerySubmit}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query"
            required
          />
          <button type="submit">Submit</button>
        </form>
        {response && (
          <div className="response">
            <h4>Response:</h4>
            <p>{response}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;