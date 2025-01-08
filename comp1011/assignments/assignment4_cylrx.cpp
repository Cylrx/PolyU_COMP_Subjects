#include <bits/stdc++.h>
using namespace std;

#define sz(x) ((int)(x).size())
#define eb emplace_back

inline void solve(int a[], int n){
	vector<int> odd, even;
	for(int i=0; i<n; i++) a[i]%2? odd.eb(a[i]) : even.eb(a[i]);
	sort(odd.begin(), odd.end());
	sort(even.begin(), even.end(), greater<int>());
	for(int i=0; i<sz(odd); i++) cout << odd[i] << " ";
	for(int i=0; i<sz(even); i++) cout << even[i] << " ";
	cout << endl;
}

void mysort(int a[], int n){
	if(n<=1) return;
	int tmp = rand() % n;
	int x = a[tmp];	
	a[tmp] = a[n-1];
	int i=0, j=n-1;
	while(i<j){
		while(i<j && a[i] < x) i++;
		if(i<j) a[j--] = a[i];
		while(i<j && a[j] >= x) j--;
		if(i<j) a[i++] = a[j];
	}
	a[i] = x;
	mysort(a, i);
	mysort(a+i+1, n-i-1);
}

inline void solve2(int a[], int n){
	int l1=0, l2=0, odd[1001], even[1001];
	for(int i=0; i<n; i++) a[i]%2? odd[l1++]=a[i] : even[l2++]=a[i];
	mysort(odd, l1);
	mysort(even, l2);
	for(int i=0; i<l1; i++) cout << odd[i] << " ";
	for(int i=l2-1; i>=0; i--) cout << even[i] << " ";
}

int main(){
	int n=0, a[1001];
	cout << "Enter a sequence of integer (-999 to finish): ";
	while(cin >> a[n++] && a[n-1]!=-999);
	//solve(a, n-1);
	solve2(a, n-1);
	return 0;
}

