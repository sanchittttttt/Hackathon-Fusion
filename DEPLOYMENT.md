# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Model Files**: Ensure your trained model is included in the repository

## Deployment Steps

### 1. Prepare Your Repository

Make sure your repository contains:
- `app.py` (main Streamlit application)
- `requirements.txt` (dependencies)
- `models/saved_models/binding_model.pt` (trained model)
- All utility files (`utils/`, `models/`, etc.)

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Set the main file path to `app.py`
6. Click "Deploy!"

### 3. Configuration

#### App URL
- Your app will be available at: `https://your-app-name.streamlit.app`

#### Environment Variables (if needed)
If you need to set environment variables:
1. Go to your app's settings
2. Add environment variables in the "Secrets" section

#### Secrets Management
For sensitive data, use Streamlit's secrets management:
1. Create `.streamlit/secrets.toml` in your repository
2. Add your secrets there
3. Access them in your app with `st.secrets["key"]`

#### Gemini AI Integration (Optional)
To enable AI-powered molecular insights:
1. Get a Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Add it to your Streamlit Cloud secrets:
   ```toml
   [api_keys]
   gemini_api_key = "your_api_key_here"
   ```
3. The app will work without it, but AI insights will be disabled

### 4. Model File Considerations

**Important**: The model file (`binding_model.pt`) is large and may cause deployment issues:

#### Option A: Include in Repository (Recommended for demo)
- Add the model file to your repository
- This works for files up to 1GB
- Simple and straightforward

#### Option B: External Storage (For production)
- Upload model to cloud storage (AWS S3, Google Cloud, etc.)
- Download model during app startup
- More complex but better for large models

### 5. Cloud Deployment Fixes

**Important**: The app has been configured to handle common cloud deployment issues:

#### RDKit Drawing Issues
- The app gracefully handles missing system libraries for molecular drawing
- If drawing fails, the app continues without molecular highlights
- No crashes due to missing `libXrender.so.1` or similar libraries

#### File Watching Issues
- Disabled file watching in `.streamlit/config.toml` to prevent inotify errors
- Set `fileWatcherType = "none"` and `fastReruns = false`
- This prevents the "inotify watch limit reached" errors

### 6. Performance Optimization

#### Caching
Your app already uses `@st.cache_data` for:
- Pipeline results caching
- Model loading optimization

#### Memory Management
- Streamlit Cloud provides 1GB RAM
- Your app is optimized for this limit
- Consider reducing `n_samples` for very large datasets

### 7. Troubleshooting

#### Common Issues:

1. **Model Loading Errors**
   - Ensure model file path is correct
   - Check file permissions
   - Verify model file is not corrupted

2. **Memory Issues**
   - Reduce batch sizes
   - Use smaller models
   - Optimize data loading

3. **Import Errors**
   - Check `requirements.txt` includes all dependencies
   - Verify package versions are compatible

4. **Slow Loading**
   - Use caching effectively
   - Optimize model loading
   - Consider lazy loading

### 8. Monitoring

Streamlit Cloud provides:
- App usage statistics
- Error logs
- Performance metrics

Access these in your app dashboard.

### 9. Updates

To update your deployed app:
1. Push changes to your GitHub repository
2. Streamlit Cloud automatically redeploys
3. Check the deployment status in your dashboard

## Example Deployment Configuration

```toml
# .streamlit/config.toml (already included)
[theme]
primaryColor = "#6366F1"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8FAFC"
textColor = "#1E293B"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
```

## Support

For deployment issues:
1. Check Streamlit Cloud documentation
2. Review your app logs
3. Test locally first
4. Contact Streamlit support if needed

---

**Your Synapse.AI app is ready for deployment! 🎉**
