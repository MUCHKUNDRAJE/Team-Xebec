import React from 'react'
import { supabase } from '../config/supabase'




function Login() {
    
    const handleGoogleLogin = async () => {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: "google",
        options: {
          redirectTo: 'http://localhost:5173/control', // or your desired URL
        },
      });
    
      if (error) console.error("Error logging in: ", error.message);
    };
  return (
    <div className='h-screen bg-[url(https://i.pinimg.com/originals/e2/52/ee/e252ee51fe694b9d14b7a00ae3f4954c.gif)]  bg-amber-100 w-full flex items-center justify-center'>

<div class="form h-96 w-96 flex items-center justify-center">
    <p className='text-white inter mb-2 font-bold'>Login</p>
  <span class=" inter title text-center">Solar Shield </span>
  <p class="description inter">Solar Shield is an AI-powered platform that predicts geomagnetic storms using real NASA space-weather data and visualizes impact zones on an interactive 3D Earth map.</p>
  <div>
    {/* <input placeholder="Enter your email" type="email" name="email" id="email-address"> */}
    <button onClick={handleGoogleLogin}>Login to Google</button>
  </div>
</div>


    </div>
  )
}

export default Login