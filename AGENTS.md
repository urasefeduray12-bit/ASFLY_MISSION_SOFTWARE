# AGENTS.md

Bu repository fiziksel bir İHA üzerinde çalışacak görev yazılımıdır. Kod değişikliklerinde temel öncelik mevcut çalışan davranışı korumak ve bilinmeyenleri artırmamaktır.

## Çalışma İlkesi

Kod üzerinde değişiklik yapmadan önce ilgili sistemi gerçekten anlamaya çalış.

Dosya isimlerine, dokümantasyona, yorumlara veya önceki açıklamalara bakarak davranış varsayma.

Gerçek davranışı mümkün olduğunca:

* çağrı zincirleri,
* importlar,
* ROS publisher/subscriber ilişkileri,
* state transition'lar,
* config kullanımı,
* runtime entry point'leri

üzerinden doğrula.

Bir şeyden emin değilsen tahmin etme.

Önce repository içinde araştır. Koddan kesin olarak çıkarılamıyorsa kullanıcıya sor.

Kullanıcının yalnızca sana verilen açıklamaları bildiğini varsayma. Kullanıcı projenin genel mimarisini ve çalışma mantığını bilir; gerektiğinde ek bilgi sağlayabilir.

## Değişiklik Yapmadan Önce

İlgili kodu ve doğrudan bağımlılıklarını oku.

Şunları anlamadan değişiklik yapma:

* Değiştirilecek dosyanın runtime rolü
* Onu kullanan veya çağıran kod
* Etkilenen topic/interface/config alanları
* Korunması gereken mevcut davranış
* İlgili testler
* Değişikliğin geri alınma yolu

## Varsayım Yapma

Özellikle şu konularda varsayım yapma:

* Bir dosyanın runtime'da aktif olup olmadığı
* Bir kodun legacy olduğu
* ROS topic bağlantıları
* State transition koşulları
* Coordinate frame yönleri
* NED/ENU dönüşümleri
* MAVLink davranışı
* Kamera yönü
* Servo/PWM değerleri
* Config değerlerinin kaynağı
* Donanım davranışı
* Gerçek saha davranışı

Repository'den doğrulanabilecek bilgiyi kullanıcıya sormadan önce araştır.

Doğrulanamayan bilgiyi uydurma.

## Minimum Değişiklik

Mevcut sistemi gereksiz yere yeniden tasarlama.

Görev gerektirmiyorsa:

* büyük refactor yapma,
* dosya taşıma,
* toplu rename yapma,
* gereksiz formatting yapma,
* dependency ekleme,
* çalışan kodu sadece daha temiz görünmesi için değiştirme.

Küçük, izole ve geri alınabilir değişiklikleri tercih et.

Görev dışında fark edilen problemleri kendiliğinden düzeltme. Ayrı bulgu olarak raporla.

## Güvenlik Kritik Alanlar

Aşağıdaki alanlarda özellikle muhafazakâr davran:

* arm/disarm
* takeoff
* velocity control
* MAVLink control
* coordinate conversion
* geofence
* failsafe
* RTL
* payload/servo kontrolü

Başka bir değişikliğin yan etkisi olarak bu alanların davranışını değiştirme.

## Test ve Doğrulama

Bu geliştirme sürecinde simülasyon ortamı yoktur.

Gazebo veya SITL:

* çalıştırılmayacak,
* test için kullanılmayacak,
* doğrulama yöntemi olarak önerilmeyecek,
* kabul kriteri yapılmayacaktır.

Doğrulamada mevcut imkanları kullan:

1. Kod akışını incele.
2. Mevcut unit/regression testlerini çalıştır.
3. Syntax ve import kontrollerini yap.
4. Interface/topic/config uyumluluğunu kontrol et.
5. Değişiklik diff'ini mevcut davranış açısından incele.
6. Uygunsa saf Python seviyesinde deterministik testler ekle.

Gerçek İHA olmadığı için donanıma bağlı davranışı doğrulanmış gibi gösterme.

Şu ayrımı koru:

* Koddan doğrulandı
* Statik olarak doğrulandı
* Test ile doğrulandı
* Gerçek İHA üzerinde doğrulandı
* Doğrulanamadı

## Agent Kullanımı

Proje agent'ları `.opencode/agents/` altında tanımlıdır.

* `chef`: araştırmayı ve değişiklik sürecini yönetir.
* `researcher`: read-only repository araştırması yapar.
* `coder`: açıkça sınırlandırılmış kod değişikliklerini uygular.

Researcher kod değiştirmez.

Coder yalnızca açık şekilde tanımlanmış kapsam içinde değişiklik yapar.

Chef gerektiğinde birden fazla researcher çağırabilir.

## Son Kural

Önce oku.

Sonra doğrula.

Varsayma.

Emin değilsen sor.

En küçük güvenli değişikliği yap.
