# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-XX

### Added
- Initial release of Clean Room Data Processor
- Multi-file upload support (CSV, Excel, JSON)
- Intelligent column analysis using Phi-3.5 Mini
- Embedding-based similarity detection
- Human-in-the-loop data cleaning workflow
- Interactive column mapping interface
- Data quality validation and anomaly detection
- Multi-format export (Excel, CSV, JSON)
- Comprehensive test suite
- Sample data files for testing

### Features
- **File Upload**: Support for CSV, Excel (.xlsx, .xls), and JSON files
- **Column Analysis**: Phi-3.5 Mini powered intelligent analysis
- **Similarity Detection**: Embedding-based column matching
- **User Feedback**: Interactive column mapping with approval workflow
- **Data Cleaning**: AI-powered transformation suggestions
- **Quality Validation**: Comprehensive data quality assessment
- **Export**: Multi-format export with metadata

### Technical
- Streamlit web interface
- Phi-3.5 Mini LLM integration
- Sentence transformers for embeddings
- Pandas for data processing
- Ollama for local LLM deployment

## [Unreleased]

### Planned
- Database connectivity (PostgreSQL, MySQL, SQLite)
- Additional file format support (Parquet, Feather)
- Batch processing for large datasets
- Custom transformation rules engine
- Real-time collaboration features
- API integration for automation
- Performance optimizations (caching, parallel processing)
- Advanced analytics and reporting 