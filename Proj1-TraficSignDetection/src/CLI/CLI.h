#pragma once

#include "ConsoleInput.h"
#include <iostream>

using std::count;

class CLI {
	private:
	

	public:
		CLI() {}
		virtual ~CLI() {}

		void startInteractiveCLI();
		void showConsoleHeader();
		int getUserOption();

		void showVersion();
};

