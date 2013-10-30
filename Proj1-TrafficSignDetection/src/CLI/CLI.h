#pragma once

#include <iostream>
#include <string>

#include "ConsoleInput.h"
#include "../ImageAnalysis/ImageAnalysis.h"

using std::count;
using std::cerr;
using std::string;


class CLI {
	public:
		CLI() {}
		virtual ~CLI() {}

		void startInteractiveCLI();
		void showConsoleHeader();
		int getUserOption();
		
		void showVersion();
};
