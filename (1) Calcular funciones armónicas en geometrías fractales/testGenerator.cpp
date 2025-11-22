#include<bits/stdc++.h>
#include<fstream>
using namespace std;

typedef complex<float> cf;
typedef complex<double> cd;
#define all(x) x.begin(),x.end()

// hacer cosas random bien hechas
mt19937_64 MT64;
void pre_init() {
	MT64 = mt19937_64(chrono::system_clock::now().
	time_since_epoch().count());
}

template<class T>
void writeFile(string fileName, vector<vector<T>> datos){
	int n = datos.size();
	if(n == 0) return;
	int m = datos[0].size();
	ofstream output(fileName);
	for(int i = 0; i < n; i++){
		for(int j = 0; j < m; j++){
			output << datos[i][j];
			if(j != m-1) output << ", ";
		}
		output << endl;
	}
	output.close();
}

vector<vector<cd>> readComplex(string fileName, int n){
	vector<vector<cd>> res(n,vector<cd>(n,0));
	ifstream lectura(fileName);
	string line; 
	for(int i = 0; i < n; i++){
		int j = 0;
		getline (lectura, line);
		stringstream ss(line);
		double a, b;
		char c;
		while(ss >> a){
			b = 0;
			c = ss.peek();
			if(c == '+'){
				ss >> c;
				ss >> b;
				ss >> c;
			}
			else if(c == '-'){
				ss >> c;
				ss >> b;
				b *= -1;
				ss >> c;
			}
			else if(c == 'i'){
				b = a;
				a = 0;
				ss >> c;
			}
			res[i][j] = cd(a,b);
			c = ss.peek();
			if(ss.peek() == ios::failbit) continue;
			else ss >> c;
			j++;
		}
	}
	lectura.close(); 
	return res;
}

int getSize(string s){
	int ans = 0;
	stringstream ss(s);
	string t;
	while(getline(ss, t, ',')){
		ans++;
	}
	return ans;
}


int main(){
	pre_init();
	cout << "Indique el tamaño de los tests (por ejemplo 101):";
	int n; cin >> n;
	if(!n%2) n++;
	int rad = (n-1)/2;
	ofstream output("circulo_rad_" + to_string(rad) + ".txt");
	rad--;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			int x = (i-rad-1);
			int y = (j-rad-1);
			if(x*x + y*y <= rad*rad) output << 0;
			else output << 1;
			if(j != n-1) output << ", ";
		}
		output << endl;
	}
	output.close();
	output.open("tipico" + to_string(n) + ".txt");
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			int x = (i-rad-1);
			int y = (j-rad-1);
			if(x*x + y*y <= rad*rad && x*x + y*y >= rad*rad/9 ) output << 0;
			else output << 1;
			if(j != n-1) output << ", ";
		}
		output << endl;
	}
	output.close();
	output.open("square" + to_string(n) + "x" + to_string(n) + ".txt");
	rad--;
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(i == 0 || j == 0 || i == n-1 || j == n-1) output << 1;
			else output << 0;
			if(j != n-1) output << ", ";
		}
		output << endl;
	}
	output.close();
	// escribir campo chiquito
	string fileName1;
	cout << "Ingrese nombre del archivo del campo complejo que desea truncar.\n";
	cin >> fileName1;
	output.open("CampoComplejo" + to_string(n) + "x" + to_string(n) + ".txt");
	ifstream lectura(fileName1);
	string line; getline (lectura, line);
	lectura.close(); 
	int N = getSize(line);
	vector<vector<cd>> campo = readComplex(fileName1, N);
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			if(campo[i][j].real() != 0)output << campo[i][j].real();
			if(campo[i][j].imag() != 0){
				if(campo[i][j].imag() > 0) output << "+";
				output << campo[i][j].imag();
				output << "i";
			}
			if(campo[i][j].real() == 0 && campo[i][j].imag() == 0) output << 0;
			if(j != n-1) output << ",";
		}
		output << endl;
	}
	output.close();
	return 0;
}

