#include "CLI.h"


void CLI::showConsoleHeader() {	
	cout << "###################################################################################################\n";
	cout << "  >>>                                    Traffic sign detection                               <<<  \n";
	cout << "###################################################################################################\n\n";
}


void CLI::startInteractiveCLI() {
	int userOption = 0;
	int cameraDeviceNumber = 0;

	do {
		ConsoleInput::getInstance()->clearConsoleScreen();
		showConsoleHeader();
		userOption = getUserOption();

		if (userOption == 3) {
			cameraDeviceNumber = ConsoleInput::getInstance()->getIntCin("  >> Insert the camera device number to use (default: 0): ", "  !!! Camera device number must be >= 0 !!!\n", 0);
		}

	} while (userOption != 0);

	cout << "\n\n\n" << endl;
	showVersion();
	cout << "\n\n" << endl;
	ConsoleInput::getInstance()->getUserInput();
}



int CLI::getUserOption() {	
	cout << " ## Detect traffic sign from:\n";
	cout << "   1 - Image\n";
	cout << "   2 - Video\n";
	cout << "   3 - Camera\n";
	cout << "   0 - Exit\n";

	return ConsoleInput::getInstance()->getIntCin("\n >>> Option [0, 3]: ", "Select one of the options above!", 0, 4);	
}



void CLI::showVersion() {
	cout << "+==================================================================================================+" << endl;
	cout << "|  Version 1.0 developed in 2013 for Computer Vision course (5th year, 1st semester, MIEIC, FEUP)  |" << endl;
	cout << "|  Author: Carlos Miguel Correia da Costa (carlos.costa@fe.up.pt / carloscosta.cmcc@gmail.com)     |" << endl;
	cout << "+==================================================================================================+" << endl;
}
