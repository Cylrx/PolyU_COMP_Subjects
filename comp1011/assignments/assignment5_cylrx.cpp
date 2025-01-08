#include <bits/stdc++.h>
using namespace std;

inline void get_max(int a[],int n){
	int r=*a;
	for(int i=1; i<n; r=r<a[i]?a[i]:r, ++i);
	cout<<"Largest Number is "<<r<<endl;
}
inline void get_min(int a[],int n){
	int r=*a;
	for(int i=1; i<n; r=r>a[i]?a[i]:r, ++i);
	cout<<"Smallest Number is "<<r<<endl;
}
inline void get_sum(int a[],int n){
	int r=0;
	for(int i=0; i<n; r+=a[i], ++i);
	cout<<"Total is "<<r<<endl;
}
inline void get_avg(int a[],int n){
	float s=0;
	for(int i=0; i<n; s+=a[i], ++i);
	cout<<"Average is "<<s/n<<endl;
}

int main(){
	cout << fixed << setprecision(3);
	ios::sync_with_stdio(0);
	cin.tie(0);
	int n=0, a[1001];
	cout << "Enter a sequence of integer (-999 to finish): " << endl;
	while(cin >> a[n++] && a[n-1]!=-999);
	if(!(n-1)) {cout << "Error: No Number" << endl; return 0;}
	get_max(a, n-1); // output largest number
	get_min(a, n-1); // output smallest number
	get_sum(a, n-1); // output the total of the numbers
	get_avg(a, n-1); // output the average of the numbers
	return 0;
}
