#include <iostream>
#include <iomanip>
#include <cmath>
#define ff first
#define ss second
#define eb emplace_back
using namespace std;

typedef pair<double, double> pdd;

inline void print_menu(){
	cout << "MENU\n";
	cout << "\t1. Divide, a/b\n";
	cout << "\t2. Multiply, a*b\n";
	cout << "\t3. Power, a^b\n";
	cout << "\t4. Square root, sqrt(a)" << endl;
}

pdd input(bool two){
	pdd x;
	cout << "input " << (two? "two" : "one") << " number" << (two? "s ": " ");
	cin >> x.ff; if(two) cin >> x.ss;
	return x;
}


void div(pdd x){!x.ss? cout << "divide by 0 error" << endl : cout << (int)x.ff << '/' << (int)x.ss << '=' << x.ff/x.ss << endl;}
void pwr(pdd x){ cout << (int)x.ff << '^' << (int)x.ss << '=' << pow(x.ff, x.ss) << endl;}
void mlt(pdd x){ cout << (int)x.ff << '*' << (int)x.ss << '=' << x.ff*x.ss << endl;}
void sqr(pdd x){ cout << "sqrt(" << (int)x.ff << ")=" << sqrt(x.ff) << endl;}


int main(){
	ios::sync_with_stdio(0);
	cout << fixed << setprecision(3);
	print_menu();
	int choice;
	cout << "Enter your choice: ";
	cin >> choice;
	switch(choice){
		case 1: div(input(1)); break;
		case 2: mlt(input(1)); break; 
		case 3: pwr(input(1)); break;
		case 4: sqr(input(0)); break;
	}
	return 0;
}
