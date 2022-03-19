import { useState, useEffect } from 'react';

const useFetch = (url) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    const performFetch = async () => {
      try {
        setLoading(true);
        const res = await fetch();
        const data = await res.json();
        setData(data);
      } catch (e) {
        setError(e);
      }
    };
    performFetch();
  }, [url]);

  return { data, loading, error };
};

export default useFetch;
