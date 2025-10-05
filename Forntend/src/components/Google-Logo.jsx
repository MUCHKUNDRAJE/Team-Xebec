import React,{useEffect ,useState} from 'react'
import { supabase } from '../config/supabase'






function Google_Logo() {

      const [user, setUser] = useState(null);

     useEffect(() => {
    const session = supabase.auth.getSession().then(({ data }) => {
      setUser(data.session?.user ?? null);
    });

    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => listener.subscription.unsubscribe();
  }, []);

  return (
    <div className='h-20 w-20 bg-red-200  absolute top-0 left-0'>
       <img src={user.user_metadata.picture} alt="" />

    </div>
  )
}

export default Google_Logo