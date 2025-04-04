import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useState, useEffect, useRef } from 'react';

const Dashboard = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [conversationHistory, setConversationHistory] = useState([]);
  const [page, setPage] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const chatContainerRef = useRef(null); // Ref to scroll to the bottom

  // Fetch conversation history when the component mounts or page changes
  useEffect(() => {
    console.log('useEffect triggered for fetching conversation history'); // Debug log
    const fetchConversationHistory = async () => {
      if (!hasMore) {
        console.log('No more history to fetch (hasMore is false)');
        return;
      }

      try {
        setLoading(true);
        const token = localStorage.getItem('token');
        console.log('Token retrieved:', token);
        if (!token) {
          console.log('No token found, redirecting to login');
          setError('No token found. Please log in again.');
          navigate('/login');
          return;
        }
        console.log('Making request to /conversation-history');
        const res = await axios.get('http://localhost:8001/conversation-history', {
          params: { limit: 50, offset: page * 50 },
          headers: { Authorization: `Bearer ${token}` },
        });
        console.log('Response from /conversation-history:', res.data);
        const newHistory = res.data.history || [];
        console.log('Fetched conversation history:', newHistory);
        if (newHistory.length < 50) setHasMore(false);
        setConversationHistory((prev) => [...prev, ...newHistory]);
        setLoading(false);
      } catch (err) {
        setError(`Error: Could not fetch conversation history. ${err.message}`);
        setLoading(false);
        console.error('Fetch error:', err);
        if (err.response?.status === 401) {
          console.log('Unauthorized, redirecting to login');
          setError('Session expired. Please log in again.');
          navigate('/login');
        }
      }
    };

    fetchConversationHistory();
  }, [page, navigate, hasMore]);

  // Scroll to the bottom of the chat container when new messages are added
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [conversationHistory]);

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        setError('No token found. Please log in again.');
        navigate('/login');
        return;
      }
      const res = await axios.post(
        'http://localhost:8001/query',
        { question: query },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      console.log('Response from backend:', res.data);
      const newResponse = res.data.response || 'No response content';

      const newEntry = {
        query: query,
        response: newResponse,
        date_created: new Date().toISOString(),
      };
      setConversationHistory((prev) => [...prev, newEntry]);
      setQuery('');
    } catch (err) {
      const errorEntry = {
        query: query,
        response: 'Error: Could not process query.',
        date_created: new Date().toISOString(),
      };
      setConversationHistory((prev) => [...prev, errorEntry]);
      console.error('Query error:', err);
      if (err.response?.status === 401) {
        setError('Session expired. Please log in again.');
        navigate('/login');
      }
    }
  };

  const handleLoadMore = () => {
    setPage((prev) => prev + 1);
  };

  return (
    <div className="dashboard" style={styles.container}>
      <h2 style={styles.header}>Dashboard</h2>
      <p style={styles.welcomeText}>Welcome, {localStorage.getItem('userEmail') || 'User'}!</p>

      {/* Conversation History Section */}
      <div style={styles.chatContainer} ref={chatContainerRef}>
        {loading && conversationHistory.length === 0 && <p style={styles.loadingText}>Loading conversation history...</p>}
        {error && <p style={styles.errorText}>{error}</p>}
        {!loading && !error && conversationHistory.length === 0 && (
          <p style={styles.placeholderText}>No conversation history yet. Ask a question to get started!</p>
        )}
        {!loading && conversationHistory.length > 0 && (
          <>
            {conversationHistory.map((entry, index) => (
              <div key={index} style={styles.chatEntry}>
                {/* User's Query (Right Side) */}
                <div style={styles.userMessage}>
                  <div style={styles.messageBubbleRight}>
                    {entry.query}
                  </div>
                </div>
                {/* Assistant's Response (Left Side) */}
                <div style={styles.assistantMessage}>
                  <div style={styles.messageBubbleLeft}>
                    {entry.response}
                  </div>
                </div>
              </div>
            ))}
            {hasMore && (
              <button
                onClick={handleLoadMore}
                style={styles.loadMoreButton}
                disabled={loading}
              >
                {loading ? 'Loading...' : 'Load More'}
              </button>
            )}
          </>
        )}
      </div>

      {/* Query Submission Section */}
      <div style={styles.queryContainer}>
        <form onSubmit={handleQuerySubmit} style={styles.form}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query"
            required
            style={styles.input}
          />
          <button type="submit" style={styles.submitButton}>Submit</button>
        </form>
      </div>
    </div>
  );
};

// Styles updated to match the latest image
const styles = {
  container: {
    maxWidth: '600px',
    margin: '0 auto',
    padding: '20px',
    backgroundColor: '#fff',
    borderRadius: '10px',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
    minHeight: '80vh',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    textAlign: 'center',
    marginBottom: '10px',
    color: '#333',
  },
  welcomeText: {
    textAlign: 'center',
    marginBottom: '20px',
    color: '#555',
  },
  chatContainer: {
    flex: 1,
    maxHeight: '60vh',
    overflowY: 'auto',
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    marginBottom: '20px',
  },
  chatEntry: {
    marginBottom: '20px',
  },
  userMessage: {
    display: 'flex',
    justifyContent: 'flex-end',
    marginBottom: '10px',
  },
  assistantMessage: {
    display: 'flex',
    justifyContent: 'flex-start',
    marginBottom: '10px',
  },
  messageBubbleRight: {
    backgroundColor: '#007bff',
    color: 'white',
    padding: '10px 15px',
    borderRadius: '20px',
    maxWidth: '60%',
    wordWrap: 'break-word',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
  },
  messageBubbleLeft: {
    backgroundColor: '#e9ecef',
    color: '#333',
    padding: '10px 15px',
    borderRadius: '20px',
    maxWidth: '60%',
    wordWrap: 'break-word',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
  },
  loadMoreButton: {
    display: 'block',
    margin: '10px auto',
    padding: '8px 16px',
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    borderRadius: '20px',
    cursor: 'pointer',
    fontWeight: 'bold',
  },
  queryContainer: {
    marginTop: 'auto',
    padding: '10px',
    backgroundColor: '#fff',
    borderRadius: '8px',
  },
  form: {
    display: 'flex',
    gap: '10px',
  },
  input: {
    flex: 1,
    padding: '10px',
    border: '1px solid #ddd',
    borderRadius: '20px',
    fontSize: '16px',
    backgroundColor: '#fff',
    color: '#333',
  },
  submitButton: {
    padding: '10px 20px',
    backgroundColor: '#007bff',
    color: 'white',
    border: 'none',
    borderRadius: '20px',
    cursor: 'pointer',
    fontWeight: 'bold',
  },
  loadingText: {
    textAlign: 'center',
    color: '#888',
  },
  errorText: {
    textAlign: 'center',
    color: 'red',
  },
  placeholderText: {
    textAlign: 'center',
    color: '#888',
  },
};

export default Dashboard;