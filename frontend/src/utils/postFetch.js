import { BACKEND_URL } from '../constants';

const postRequst = async (realtiveUrl, data) => {
  const url = `${BACKEND_URL}/${realtiveUrl}`;
  const res = fetch(url, {
    method: 'POST',
    headers: {
      'Content-type': 'application/json',
    },
    body: JSON.stringify(data),
  });
  return await res.json();
};

export default postRequst;
