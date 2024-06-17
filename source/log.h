#pragma once


#if defined(DEBUG) || defined(_DEBUG)
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"
std::shared_ptr<spdlog::logger>& GetLogger();

#define FGAC_LOG_TRACE(...)    GetLogger()->trace(__VA_ARGS__)
#define FGAC_LOG_INFO(...)     GetLogger()->info(__VA_ARGS__)
#define FGAC_LOG_WARN(...)     GetLogger()->warn(__VA_ARGS__)
#define FGAC_LOG_ERROR(...)    GetLogger()->error(__VA_ARGS__)
#define FGAC_LOG_CRITICAL(...) GetLogger()->critical(__VA_ARGS__)



#else

#define FGAC_LOG_TRACE(...)    
#define FGAC_LOG_INFO(...)     
#define FGAC_LOG_WARN(...)     
#define FGAC_LOG_ERROR(...)    
#define FGAC_LOG_CRITICAL(...) 

#endif